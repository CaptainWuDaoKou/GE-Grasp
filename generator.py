import numpy as np
# import cv2
# import time
import random

# Set parameters
finger_width = 0.02
heightmap_resolution = 0.002
gripper_open_width_inner = 0.12
gripper_open_width_outter = 0.14
num_det_samples = int(finger_width / heightmap_resolution)


def get_pointcloud(heightmap_resolution, valid_depth_heightmap, workspace_limits):

    points = np.zeros((valid_depth_heightmap.shape[0], valid_depth_heightmap.shape[1], 3))
    for x in range(valid_depth_heightmap.shape[0]):
        for y in range(valid_depth_heightmap.shape[1]):
            points[x][y] = [x * heightmap_resolution + workspace_limits[0][0],
                            y * heightmap_resolution + workspace_limits[1][0],
                            valid_depth_heightmap[y][x] + workspace_limits[2][0]]
    pointcloud = points.reshape(-1, 3)
    points = points.reshape((-1, 3))
    with open('point_cloud.txt', 'w') as f:
        for i in range(len(points)):
            f.write(str(np.float(points[i][0])) + ';' +
                    str(np.float(points[i][1])) + ';' +
                    str(np.float(points[i][2])) + '\n'
                    )
    return pointcloud



def grasp_generator(target_center, pointcloud, valid_depth_heightmap):

    pointcloud_reshaped = pointcloud.reshape((224, 224, 3)) # correct
    # area_shape_default = [112, 112]
    area_shape_default = [100, 100]

    target_center_x = target_center[0] 
    target_center_y = target_center[1]

    proposal_area_x = np.arange(max(0, target_center_x - area_shape_default[0]/2), min(target_center_x + area_shape_default[0]/2, 223)) 
    proposal_area_y = np.arange(max(0, target_center_y - area_shape_default[1]/2), min(target_center_y + area_shape_default[1]/2, 223))

    # Get the indices of the area of interest
    area_indices = np.zeros((len(proposal_area_x), len(proposal_area_y), 2), dtype=np.int) 
    for i, x in enumerate(proposal_area_x):
        for j, y in enumerate(proposal_area_y):
            area_indices[i, j] = [x, y]
    area_indices = area_indices.reshape((-1, 2)) 

    # Get translation matrices of height detection points
    grasp_ind_mat = area_indices.copy()
    hshift = 6
    vshift = 24
    finger_thickness = 10
    det_inds_mat = np.zeros((5*3, len(proposal_area_x)*len(proposal_area_y), 2))
    det_height_mat = np.zeros((5*3, len(proposal_area_x)*len(proposal_area_y)))
    translations = np.zeros((5, 3, 2))

    for i in range(translations.shape[0]):
        for j in range(translations.shape[1]):
            translations[i, j] = [(j - 1) * hshift, (i - 2) * vshift]  # x, y
    translations[0, :, 1] = translations[1, :, 1] - finger_thickness
    translations[4, :, 1] = translations[3, :, 1] + finger_thickness
    # Get valid grasp point indices with different rotations
    grasps = []
    for k in range(16):
        theta = np.pi / 16 * k
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        translations_rot = np.matmul(translations, R).astype(np.int)

        # Get detection point indices with different rotations
        for i in range(5):
            for j in range(3):
                det_inds_mat[3*i+j, :] = np.floor((grasp_ind_mat + translations_rot[i, j]))
        det_inds_mat = det_inds_mat.astype(np.int)

        # Get height data of detection points
        for i in range(5):
            for j in range(3):
                idx = tuple([det_inds_mat[3*i+j, :, 1], det_inds_mat[3*i+j, :, 0]])
                idx[0][idx[0] > 223] = 223
                idx[0][idx[0] < 0] = 0
                idx[1][idx[1] > 223] = 223
                idx[1][idx[1] < 0] = 0
                det_height_mat[3*i+j, :] = valid_depth_heightmap[idx]

        height_det_grasp = np.vstack((det_height_mat[6], det_height_mat[7], det_height_mat[8])) # height of the potential grasp point area
        height_det1 = np.vstack((det_height_mat[0], det_height_mat[1], det_height_mat[2], det_height_mat[3], det_height_mat[4], det_height_mat[5])) # height of the left finger area
        height_det2 = np.vstack((det_height_mat[9], det_height_mat[10], det_height_mat[11], det_height_mat[12], det_height_mat[13], det_height_mat[14])) # height of the right finger area

        height_grasp_min = np.min(height_det_grasp, axis=0)
        height_det1_max = np.max(height_det1, axis=0)
        height_det2_max = np.max(height_det2, axis=0)

        valid = (height_grasp_min - height_det2_max > 0.025) & (height_grasp_min - height_det1_max > 0.025)
        valid_grasp_ind = np.where(valid==True)[0].flatten()

        valid_grasp_inds = area_indices[valid_grasp_ind] # Get the indices of the valid grasps

        grasps.append([k, pointcloud_reshaped[tuple([valid_grasp_inds[:, 0], valid_grasp_inds[:, 1]])], valid_grasp_inds])


    num_valid_grasps = 0
    for i in range(len(grasps)):
        num_valid_grasps += len(grasps[i][1])

    grasp_mask_heightmaps = []
    for rot_idx in np.array(grasps)[:, 0]:
        for grasp_ind in np.array(grasps)[rot_idx, 2]:
            heightmap = grasp_mask_generator(rot_idx, grasp_ind)
            grasp_mask_heightmaps.append([heightmap, grasp_ind, rot_idx])
    grasp_mask_heightmaps = np.asarray(grasp_mask_heightmaps)

    return grasps, grasp_mask_heightmaps, num_valid_grasps


def grasp_mask_generator(rot_idx, grasp_ind):
    hshift = 8
    # vshift = 24
    vshift = 31

    theta = np.pi / 16 * rot_idx
    grasp_mask_heightmap = np.zeros((224, 224), dtype='uint8')
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    trans_x, trans_y = np.mgrid[-hshift: hshift+1: 1, -vshift: vshift+1: 1]
    tras_indices = np.c_[trans_x.ravel(), trans_y.ravel()]
    trans_rot_indices = np.floor(np.matmul(tras_indices, R)).astype(np.int)
    grasp_indices = trans_rot_indices + grasp_ind
    grasp_indices[grasp_indices > 223] = 223
    grasp_indices[grasp_indices < 0] = 0

    grasp_mask_heightmap[tuple([grasp_indices[:, 1], grasp_indices[:, 0]])] = 1

    return grasp_mask_heightmap


def push_mask_generator(push_start_point, target_center, rot_ind):

    push_mask = np.zeros((224, 224))
    translations = np.zeros((50, 10, 2)).astype(np.int)
    if push_start_point[1] < target_center[1]:
        for i in range(0, 10):
            for j in range(0, 50):
                translations[j, i, :] = [i - 5, j]
    else:
        for i in range(0, 10):
            for j in range(0, 50):
                translations[j, i, :] = [i - 5, -j]

    push_vec = [target_center[0] - push_start_point[0], target_center[1] - push_start_point[1] + 1e-8]
    push_rot_angle = np.arctan(push_vec[0] / push_vec[1])
    rot_angle = np.pi / 8 * (rot_ind - 1)
    rot_angle_final = push_rot_angle + rot_angle
    rot_mat = np.array([[np.cos(rot_angle_final), -np.sin(rot_angle_final)],
                        [np.sin(rot_angle_final), np.cos(rot_angle_final)]])

    translations_rot = np.dot(translations, rot_mat).astype(np.int)
    push_area_inds = push_start_point + translations_rot
    push_area_inds[np.where(push_area_inds > 223)] = 223
    push_area_inds[np.where(push_area_inds < 0)] = 0
    push_mask[tuple([push_area_inds[0: 25, :, 1], push_area_inds[0: 25, :, 0]])] = 0.5
    push_mask[tuple([push_area_inds[25: 50, :, 1], push_area_inds[25: 50, :, 0]])] = 1

    return push_mask


def push_generator(target_center, valid_depth_heightmap):

    area_shape_default = [100, 100]
    height_target = valid_depth_heightmap[target_center[1], target_center[0]]

    target_center_x = target_center[0]
    target_center_y = target_center[1]

    proposal_area_x = np.arange(max(0, target_center_x - area_shape_default[0] / 2),
                                min(target_center_x + area_shape_default[0] / 2, 223))
    proposal_area_y = np.arange(max(0, target_center_y - area_shape_default[1] / 2),
                                min(target_center_y + area_shape_default[1] / 2, 223))

    # Get the indices of the area of interest
    area_indices = np.zeros((len(proposal_area_x), len(proposal_area_y), 2), dtype=np.int)
    for i, x in enumerate(proposal_area_x):
        for j, y in enumerate(proposal_area_y):
            area_indices[i, j] = [x, y]
    area_indices = area_indices.reshape((-1, 2))  

    # Searching for suitable starting points as candidates
    det_inds_mat = np.zeros((4, len(proposal_area_x) * len(proposal_area_y), 2))
    det_height_mat = np.zeros((4, len(proposal_area_x) * len(proposal_area_y)))
    push_ind_mat = area_indices.copy()
    vshift = 6
    hshift = 6
    translations = np.zeros((2, 2, 2))
    for i in range(translations.shape[0]):
        for j in range(translations.shape[1]):
            translations[i, j] = [(2*j - 1) * hshift, (2*i - 1) * vshift]
    for i in range(2):
        for j in range(2):
            det_inds_mat[2 * i + j, :] = np.floor((push_ind_mat + translations[i, j]))
    det_inds_mat = det_inds_mat.astype(np.int)

    for i in range(2):
        for j in range(2):
            idx = tuple([det_inds_mat[2 * i + j, :, 1], det_inds_mat[2 * i + j, :, 0]])
            idx[0][idx[0] > 223] = 223
            idx[0][idx[0] < 0] = 0
            idx[1][idx[1] > 223] = 223
            idx[1][idx[1] < 0] = 0
            det_height_mat[2 * i + j, :] = valid_depth_heightmap[idx]

    height_det = np.max(det_height_mat, axis=0)
    valid = (height_target - height_det >= 0.015)
    valid_push_inds = np.where(valid == True)[0].flatten()
    num_valid_push = valid_push_inds.shape[0]

    if num_valid_push >= 1:
        quadrants = [[], [], [], []]
        valid_push_ids = area_indices[valid_push_inds]
        for idx in valid_push_ids:
            if idx[0] <= target_center[0] and idx[1] <= target_center[1]:
                quadrants[0].append(idx)
            elif idx[0] > target_center[0] and idx[1] <= target_center[1]:
                quadrants[1].append(idx)
            elif idx[0] <= target_center[0] and idx[1] > target_center[1]:
                quadrants[2].append(idx)
            elif idx[0] > target_center[0] and idx[1] > target_center[1]:
                quadrants[3].append(idx)

        candidates = []
        for sector in quadrants:
            if len(sector) > 10:
                sampled_push = random.sample(sector, 10)
                candidates.append(sampled_push)
            else:
                candidates.append(sector)

        pushes = []
        for sector in candidates:
            for start_point in sector:
                push = []
                rot_ind = np.random.randint(0, 3)
                push_mask = push_mask_generator(start_point, target_center, rot_ind)
                push.append(start_point)
                push.append(rot_ind)
                push.append(push_mask)
                pushes.append(push)

    else:
        if target_center[0] < 112:
            push_start_point = [np.max((target_center[0] - 25, 0)), target_center[1]]
        else:
            push_start_point = [np.min((target_center[0] + 25, 223)), target_center[1]]
        rot_ind = np.random.randint(0, 3)
        push_mask = push_mask_generator(push_start_point, target_center, rot_ind)
        pushes = [[push_start_point, rot_ind, push_mask]]

    return pushes

