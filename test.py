#!/usr/bin/env python
import time
import datetime
import os
import argparse
import torch
import numpy as np
import cv2

from robot import Robot
from lwrf_infer import LwrfInfer
from evaluator import Evaluator
from logger import Logger
from generator import grasp_generator, push_generator, get_pointcloud
import utils
global sample_iteration


def main(args):
    threshold = 1.0
    trials_per_case = 30
    
    test_preset_cases = args.test_preset_cases
    if test_preset_cases:
        cases_list = ['simulation/preset/coordination-00.txt',
                      'simulation/preset/coordination-01.txt',
                      'simulation/preset/coordination-02.txt',
                      'simulation/preset/coordination-03.txt',
                      'simulation/preset/coordination-04.txt',
                      'simulation/preset/coordination-05.txt',
                      'simulation/preset/coordination-06.txt',
                      'simulation/preset/coordination-07.txt']
    else:
        cases_list = ['simulation/random/random-3blocks.txt',
                      'simulation/random/random-8blocks.txt',
                      'simulation/random/random-13blocks.txt']

    # --------------- Setup options ---------------
    workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]])# Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
    heightmap_resolution = args.heightmap_resolution
    random_seed = args.random_seed
    force_cpu = args.force_cpu

    # -------------- Testing options --------------
    is_testing = True
    test_target_seeking = args.test_target_seeking
    max_test_trials = args.max_test_trials  # Maximum number of test runs per case/scenario
    max_motion_onecase = args.max_motion_onecase

    # ------ Pre-loading and logging options ------
    load_ckpt = args.load_ckpt  # Load pre-trained ckpt of model
    critic_ckpt_file = os.path.abspath(args.critic_ckpt) if load_ckpt else None
    continue_logging = args.continue_logging  # Continue logging from previous session
    save_visualizations = args.save_visualizations

    # ------ Initialize some status variables -----
    seeking_target = False,
    margin_occupy_ratio = None,
    margin_occupy_norm = None,
    best_grasp_pix_ind = None,
    best_pix_ind = None,
    grasp_succeeded = False,
    grasp_effective = False,
    target_grasped = False

    print('-----------------------------------')
    # Set random seed
    np.random.seed(random_seed)
    # Initialize evaluator
    evaluator = Evaluator(0.5, is_testing, load_ckpt, critic_ckpt_file, force_cpu)
    
    grasp_fail_count = [0]
    motion_fail_count = [0]

    # Reposition objects
    def reposition_objects():
        robot.restart_sim()
        robot.add_objects()
        grasp_fail_count[0] = 0
        motion_fail_count[0] = 0
    
    avg_actions = []
    for config_file in cases_list:

        # Initialize robot system
        actions_count_scene = []
        robot = Robot(workspace_limits, is_testing, test_preset_cases, config_file)
        logging_directory = os.path.join(os.path.abspath('logs'), 'testing/release', config_file.split('/')[-1].split('.')[0])
        logger = Logger(logging_directory)
        # Define light weight refinenet model
        lwrf_model = LwrfInfer(use_cuda=evaluator.use_cuda, save_path=logger.lwrf_results_directory)
        target_name = None
        counter_scene = 0
        prev_grasp_fail_count = [0]

        while counter_scene < trials_per_case:  
            if test_target_seeking and target_grasped:
                reposition_objects()
                target_name = None
                del prev_color_img
            print('\n%s iteration: %d' % ('Testing' if is_testing else 'Training', evaluator.iteration))
            iteration_time_0 = time.time()

            # Make sure simulation is still stable (if not, reset simulation)
            robot.check_sim()

            # Get latest RGB-D image
            color_img, depth_img = robot.get_camera_data()
            depth_img = depth_img * robot.cam_depth_scale  # Apply depth scale from calibration

            # Use lwrf to segment/detect target object
            segment_results = lwrf_model.segment(color_img)

            # Get heightmap from RGB-D image (by re-projecting 3D point cloud)
            color_heightmap, depth_heightmap, seg_mask_heightmaps = utils.get_heightmap(
                color_img, depth_img, segment_results['masks'], robot.cam_intrinsics, robot.cam_pose, workspace_limits,
                heightmap_resolution) 

            valid_depth_heightmap = depth_heightmap.copy()
            valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0

            mask_heightmaps = utils.process_mask_heightmaps(segment_results, seg_mask_heightmaps) 

            # Check targets
            if (len(mask_heightmaps['names']) == 0 and not test_target_seeking) or motion_fail_count[0] >= max_motion_onecase:
                # Restart if no targets detected
                counter_scene += 1
                reposition_objects()
                target_name = None
                continue

            # Choose target
            if len(mask_heightmaps['names']) == 0 and test_target_seeking:
                seeking_target = True
                target_mask_heightmap = np.ones_like(valid_depth_heightmap)
            else:
                seeking_target = False
                if target_name in mask_heightmaps['names']:
                    target_id = mask_heightmaps['names'].index(target_name)
                    target_mask_heightmap = mask_heightmaps['heightmaps'][target_id]
                else:
                    target_id = 0  # Set the first object in the segmentation result as the target
                    target_name = mask_heightmaps['names'][target_id]
                    target_mask_heightmap = mask_heightmaps['heightmaps'][target_id]

                y, x = np.where(target_mask_heightmap==1.0)
                target_center = (int(np.mean(x)), int(np.mean(y)))
                print('Target name:', target_name)
                margin_occupy_ratio, margin_occupy_norm = utils.check_grasp_margin(target_mask_heightmap, depth_heightmap)

            # Generate grasp and push candidates
            point_cloud = get_pointcloud(heightmap_resolution, valid_depth_heightmap, workspace_limits)
            point_cloud_reshaped = point_cloud.reshape((224, 224, -1))
            target_position = point_cloud_reshaped[target_center]

            # Choose best push
            pushes = push_generator(target_center, valid_depth_heightmap)
            evaluator.model.load_state_dict(torch.load('saved_models/pushing.pkl'))

            push_confs = []
            for push in pushes:
                push_start_point = push[0]
                rot_ind = push[1]
                push_mask = push[2]

                depth_heightmap = np.asarray(valid_depth_heightmap * 370, dtype=np.int) # ***
                target_mask = target_mask_heightmap * 255
                push_mask_input = push_mask * 255

                confidence, _ = evaluator.forward(depth_heightmap, target_mask, push_mask_input)
                push_confs.append(confidence.item())
            print('best push value: ', np.max(push_confs))
            best_push_ind = np.argmax(push_confs)
            best_push_mask = pushes[best_push_ind][2]
            best_push_start_point = pushes[best_push_ind][0]
            best_push_rot_ind = pushes[best_push_ind][1]
            push_start_position = point_cloud_reshaped[best_push_start_point[0], best_push_start_point[1]]

            # Choose best grasp
            grasps, grasp_mask_heightmaps, num_grasps = grasp_generator(target_center, point_cloud, valid_depth_heightmap)
            evaluator.model.load_state_dict(torch.load('saved_models/grasping.pkl'))

            ########### Coordinating between pushing and grasping ###########
            if num_grasps == 0:
                primitive_action = 'push'
                continue
            elif num_grasps > 100:
                sampled_inds = np.random.choice(np.arange(num_grasps), 100, replace=False)
            else:
                sampled_inds = np.random.choice(np.arange(num_grasps), num_grasps, replace=False)

            if num_grasps > 0:
                confs, grasp_inds, rot_inds = [], [], []
                grasp_masks = []
                for i in sampled_inds:
                    grasp_mask_heightmap = grasp_mask_heightmaps[i][0]

                    depth_heightmap = np.asarray(valid_depth_heightmap * 370, dtype=np.int) # ***
                    target_mask = target_mask_heightmap * 255
                    grasp_mask = grasp_mask_heightmap * 255

                    confidence, _ = evaluator.forward(depth_heightmap, target_mask, grasp_mask)

                    confs.append(confidence.item())
                    grasp_inds.append(grasp_mask_heightmaps[i][1])
                    rot_inds.append(grasp_mask_heightmaps[i][2])
                    grasp_masks.append(grasp_mask_heightmaps[i][0])

                grasp_inds = np.hstack((np.array(rot_inds).reshape((-1, 1)), np.array(grasp_inds)))
                grasp_masks = np.array(grasp_masks)

                best_grasp_conf = np.max(confs)
                best_grasp_ind = np.argmax(confs)
                best_grasp_pix_ind = grasp_inds[best_grasp_ind]
                best_grasp_mask = grasp_masks[best_grasp_ind]

            else:
                best_grasp_conf = 0
                primitive_action = 'push'

            ############## Executing action ########################
            if best_grasp_conf < threshold:  
                primitive_action = 'push'
                best_pix_ind = [best_push_rot_ind, best_push_start_point[0], best_push_start_point[1]]
            else:
                primitive_action = 'grasp'
                best_pix_ind = best_grasp_pix_ind

            if prev_grasp_fail_count[0] >= 2:
                primitive_action = 'push'
                grasp_fail_count[0] = 0

            # Compute 3D position of pixel
            best_rotation_angle = - (np.deg2rad(best_pix_ind[0] * (180.0 / evaluator.model.num_rotations)) + np.pi / 2)
            best_pix_x = best_pix_ind[1]
            best_pix_y = best_pix_ind[2]
            primitive_position = [best_pix_x * heightmap_resolution + workspace_limits[0][0],
                                  best_pix_y * heightmap_resolution + workspace_limits[1][0],
                                  valid_depth_heightmap[best_pix_y][best_pix_x] + workspace_limits[2][0]]

            # If pushing, adjust start position, and make sure z value is safe and not too low
            if primitive_action == 'push':
                finger_width = 0.02
                safe_kernel_width = int(np.round((finger_width / 2) / heightmap_resolution))
                local_region = valid_depth_heightmap[
                               max(best_pix_y - safe_kernel_width, 0):min(best_pix_y + safe_kernel_width + 1,
                                                                          valid_depth_heightmap.shape[0]),
                               max(best_pix_x - safe_kernel_width, 0):min(best_pix_x + safe_kernel_width + 1,
                                                                          valid_depth_heightmap.shape[1])]
                if local_region.size == 0:
                    safe_z_position = workspace_limits[2][0]
                else:
                    safe_z_position = np.max(local_region) + workspace_limits[2][0]
                primitive_position[2] = safe_z_position

            # Initialize variables that influence reward
            target_grasped = False
            grasp_succeeded = False

            # Executing action
            motion_fail_count[0] += 1
            print('best grasp value: ', best_grasp_conf)

            if primitive_action == 'push':           
                if target_center[0] < 45 or target_center[0] > 179 or target_center[1] < 45 or target_center[1] > 179:
                    robot.retrieval(target_position, workspace_limits)
                else:
                    robot.push(push_start_position, best_push_rot_ind, target_position, workspace_limits) 

            elif primitive_action == 'grasp':
                grasp_fail_count[0] += 1
                grasp_succeeded, grasped_object_name = robot.grasp(primitive_position, best_rotation_angle,
                                                                   workspace_limits)  
                if grasp_succeeded:
                    print('Grasping succeeded, Grasped ', grasped_object_name)
                    target_grasped = grasped_object_name == target_name
                    print('Target grasped?:', target_grasped)
                    if target_grasped:
                        actions_count_scene.append(motion_fail_count[0])
                        print('actions_count_scene', actions_count_scene)
                        print('mean actions_count_scene', np.mean(actions_count_scene))
                        motion_fail_count[0] = 0
                        grasp_fail_count[0] = 0
                else:
                    print('Grasping failed')
                    grasp_succeeded = False
                    target_grasped = False

            # -------------------------------------------------------------
            if 'prev_color_img' in locals():
                margin_increased = False
                blockage_decreased = False

                if not robot.objects_reset:
                    if not prev_target_grasped:
                        # Detect whether the target edge area increased
                        margin_increase_threshold = 0.1
                        margin_increase_val = prev_margin_occupy_ratio - margin_occupy_ratio
                        if margin_increase_val > margin_increase_threshold:
                            margin_increased = True
                            print('Grasp margin increased: (value: %f)' % margin_increase_val)

                        # Detect whether the blockage over target decreased
                        current_target_mask_area = sum(sum(target_mask_heightmap == 1))
                        prev_target_mask_area = sum(sum(prev_target_mask_heightmap == 1))
                        blockage_decreased_threshold = 0.25
                        blockage_decreased_ratio = (current_target_mask_area - prev_target_mask_area) / prev_target_mask_area
                        if blockage_decreased_ratio > blockage_decreased_threshold:
                            blockage_decreased = True
                            print('Target blockage decreased: (value: %.2f)' % blockage_decreased_ratio)
                grasp_effective = (blockage_decreased or margin_increased) and grasp_succeeded
                env_change_detected, _ = utils.check_env_depth_change(prev_depth_heightmap, depth_heightmap)
                motion_target_oriented = utils.check_grasp_target_oriented(prev_best_pix_ind, prev_target_mask_heightmap)

            # -------------------------------------------------------------

            # Save information for next training step
            if not seeking_target:
                prev_color_img = color_img.copy()
                prev_depth_img = depth_img.copy()
                prev_color_heightmap = color_heightmap.copy()
                prev_depth_heightmap = depth_heightmap.copy()
                prev_valid_depth_heightmap = valid_depth_heightmap.copy()

                prev_mask_heightmaps = mask_heightmaps.copy()
                prev_target_mask_heightmap = target_mask_heightmap.copy()
                prev_grasp_mask_heightmap = grasp_mask_heightmap.copy()
                prev_push_mask_heightmap = push_mask.copy()

                prev_target_grasped = target_grasped
                prev_grasp_succeeded = grasp_succeeded
                prev_grasp_effective = grasp_effective
                prev_primitive_action = primitive_action
                prev_best_pix_ind = best_pix_ind
                prev_margin_occupy_ratio = margin_occupy_ratio

                prev_grasp_fail_count = grasp_fail_count.copy()

            robot.objects_reset = False
            evaluator.iteration += 1
            iteration_time_1 = time.time()
            print('Time elapsed: %f' % (iteration_time_1-iteration_time_0))
        avg_actions.append(np.mean(actions_count_scene))

    print('final result:', avg_actions, np.mean(avg_actions))


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()

    # --------------- Setup options ---------------
    parser.add_argument('--heightmap_resolution', dest='heightmap_resolution', type=float, action='store',
                        default=0.002, help='meters per pixel of heightmap')
    parser.add_argument('--random_seed', dest='random_seed', type=int, action='store', default=1234)
    parser.add_argument('--force_cpu', dest='force_cpu', action='store_true', default=False)

    # -------------- Testing options --------------
    parser.add_argument('--is_testing', dest='is_testing', action='store_true', default=False)
    parser.add_argument('--test_preset_cases', dest='test_preset_cases', action='store_true', default=False)
    parser.add_argument('--test_target_seeking', dest='test_target_seeking', action='store_true', default=False)
    parser.add_argument('--max_motion_onecase', dest='max_motion_onecase', type=int, action='store', default=20,
                        help='maximum number of motions per test trial')
    parser.add_argument('--max_test_trials', dest='max_test_trials', type=int, action='store', default=5,
                        help='number of repeated test trials')

    # ------ Pre-loading and logging options ------
    parser.add_argument('--load_ckpt', dest='load_ckpt', action='store_true', default=False)
    parser.add_argument('--critic_ckpt', dest='critic_ckpt', action='store')
    parser.add_argument('--coordinator_ckpt', dest='coordinator_ckpt', action='store')
    parser.add_argument('--continue_logging', dest='continue_logging', action='store_true', default=False)
    parser.add_argument('--logging_directory', dest='logging_directory', action='store')
    parser.add_argument('--save_visualizations', dest='save_visualizations', action='store_true', default=True)

    # Run main program with specified arguments
    args = parser.parse_args()
    main(args)








