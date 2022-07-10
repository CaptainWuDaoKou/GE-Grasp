import os
import numpy as np
# import cv2
import torch
# from torch.autograd import Variable
from models import evaluator_net
from scipy import ndimage
import collections


class Evaluator(object):
    def __init__(self, future_reward_discount, is_testing, load_snapshot, snapshot_file, force_cpu):

        # Check if CUDA can be used
        if torch.cuda.is_available() and not force_cpu:
            print("CUDA detected. Running with GPU acceleration.")
            self.use_cuda = True
        elif force_cpu:
            print("CUDA detected, but overriding with option '--cpu'. Running with only CPU.")
            self.use_cuda = False
        else:
            print("CUDA is *NOT* detected. Running with only CPU.")
            self.use_cuda = False

        # Fully convolutional Q network for deep reinforcement learning
        self.model = evaluator_net(self.use_cuda)
        self.future_reward_discount = future_reward_discount
        # self.num_rotations = 16

        # Initialize Huber loss
        # self.criterion = torch.nn.SmoothL1Loss(reduction='none')  # Huber loss
        self.criterion = torch.nn.L1Loss()

        if self.use_cuda:
            self.criterion = self.criterion.cuda()

        # Load pre-trained model
        if load_snapshot:
            self.model.load_state_dict(torch.load(snapshot_file))
            print('Pre-trained RL snapshot loaded from: %s' % snapshot_file)

        # Convert model from CPU to GPU
        if self.use_cuda:
            self.model = self.model.cuda()

        # Set model to training mode
        self.model.train()

        # Initialize optimizer
        if is_testing:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0, momentum=0.9, weight_decay=0.0)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.9, weight_decay=2e-5)

        self.iteration = 0

        # Initialize lists to save execution info and RL variables
        self.executed_action_log = []
        self.label_value_log = []
        self.reward_value_log = []
        self.predicted_value_log = []
        self.reposition_log = []
        self.augment_ids = []
        self.target_grasped_log = []
        self.loss_queue = collections.deque([], 10)
        self.loss_rec = []
        self.sync_loss = []
        self.sync_acc = []

    # Pre-load execution info and RL variables
    def preload(self, transitions_directory):
        self.executed_action_log = np.loadtxt(os.path.join(transitions_directory, 'executed-action.log.txt'),
                                              delimiter=' ')
        self.iteration = self.executed_action_log.shape[0] - 2
        self.executed_action_log = self.executed_action_log[0:self.iteration, :]
        self.executed_action_log = self.executed_action_log.tolist()
        self.label_value_log = np.loadtxt(os.path.join(transitions_directory, 'label-value.log.txt'), delimiter=' ')
        self.label_value_log = self.label_value_log[0:self.iteration]
        self.label_value_log.shape = (self.iteration, 1)
        self.label_value_log = self.label_value_log.tolist()
        self.predicted_value_log = np.loadtxt(os.path.join(transitions_directory, 'predicted-value.log.txt'),
                                              delimiter=' ')
        self.predicted_value_log = self.predicted_value_log[0:self.iteration]
        self.predicted_value_log.shape = (self.iteration, 1)
        self.predicted_value_log = self.predicted_value_log.tolist()
        self.reward_value_log = np.loadtxt(os.path.join(transitions_directory, 'reward-value.log.txt'), delimiter=' ')
        self.reward_value_log = self.reward_value_log[0:self.iteration]
        self.reward_value_log.shape = (self.iteration, 1)
        self.reward_value_log = self.reward_value_log.tolist()
        self.reposition_log = np.loadtxt(os.path.join(transitions_directory, 'reposition.log.txt'), delimiter=' ')
        self.reposition_log.shape = (self.reposition_log.shape[0], 1)
        self.reposition_log = self.reposition_log.tolist()
        self.augment_ids = np.loadtxt(os.path.join(transitions_directory, 'augment-ids.log.txt'), delimiter=' ')
        self.augment_ids = self.augment_ids[self.augment_ids <= self.iteration].astype(int)
        self.augment_ids = self.augment_ids.tolist()
        self.target_grasped_log = np.loadtxt(os.path.join(transitions_directory, 'target-grasped.log.txt'),
                                             delimiter=' ')
        self.target_grasped_log = self.target_grasped_log.astype(int).tolist()
        self.loss_rec = np.loadtxt(os.path.join(transitions_directory, 'loss-rec.log.txt'), delimiter=' ')
        self.loss_rec = self.loss_rec.tolist()
        self.sync_loss = np.loadtxt(os.path.join(transitions_directory, 'sync-loss.log.txt'), delimiter=' ')
        self.sync_loss = self.sync_loss.tolist()
        self.sync_acc = np.loadtxt(os.path.join(transitions_directory, 'sync-acc.log.txt'), delimiter=' ')
        self.sync_acc = self.sync_acc.tolist()

    # Compute forward pass through model to compute affordances/Q
    def forward(self, depth_heightmap, target_mask_heightmap, grasp_mask_heightmap):

        # Apply 2x scale to input heightmaps
        depth_heightmap_2x = ndimage.zoom(depth_heightmap, zoom=[2, 2], order=0)
        target_mask_heightmap_2x = ndimage.zoom(target_mask_heightmap, zoom=[2, 2], order=0)
        grasp_mask_heightmap_2x = ndimage.zoom(grasp_mask_heightmap, zoom=[2, 2], order=0)
        assert (depth_heightmap_2x.shape[0:2] == target_mask_heightmap_2x.shape[0:2])

        # Add extra padding (to handle rotations inside network)
        diag_length = float(depth_heightmap_2x.shape[0]) * np.sqrt(2)
        diag_length = np.ceil(diag_length / 32) * 32
        padding_width = int((diag_length - depth_heightmap_2x.shape[0]) / 2)
        depth_heightmap_2x = np.pad(depth_heightmap_2x, padding_width, 'constant', constant_values=0)
        target_mask_heightmap_2x = np.pad(target_mask_heightmap_2x, padding_width, 'constant', constant_values=0)
        grasp_mask_heightmap_2x = np.pad(grasp_mask_heightmap_2x, padding_width, 'constant', constant_values=0)

        # Pre-process input images
        image_mean = 0.01
        image_std = 0.03
        depth_heightmap_2x.shape = (depth_heightmap_2x.shape[0], depth_heightmap_2x.shape[1], 1)
        input_depth_image = (depth_heightmap_2x - image_mean) / image_std

        target_mask_heightmap_2x.shape = (target_mask_heightmap_2x.shape[0], target_mask_heightmap_2x.shape[1], 1)
        input_target_mask_image = target_mask_heightmap_2x

        grasp_mask_heightmap_2x.shape = (grasp_mask_heightmap_2x.shape[0], grasp_mask_heightmap_2x.shape[1], 1)
        input_grasp_mask_image = grasp_mask_heightmap_2x

        # Construct minibatch of size 1 (b,c,h,w)
        input_depth_image.shape = (input_depth_image.shape[0], input_depth_image.shape[1], input_depth_image.shape[2], 1)
        input_target_mask_image.shape = (input_target_mask_image.shape[0], input_target_mask_image.shape[1], input_target_mask_image.shape[2], 1)
        input_grasp_mask_image.shape = (input_grasp_mask_image.shape[0], input_grasp_mask_image.shape[1], input_grasp_mask_image.shape[2], 1)
        
        input_depth_data = torch.from_numpy(input_depth_image.astype(np.float32)).permute(3, 2, 0, 1)
        input_target_mask_data = torch.from_numpy(input_target_mask_image.astype(np.float32)).permute(3, 2, 0, 1)
        input_grasp_mask_data = torch.from_numpy(input_grasp_mask_image.astype(np.float32)).permute(3, 2, 0, 1)

        # Pass input data through model
        confidence, state_feat = self.model.forward(input_depth_data, input_target_mask_data, input_grasp_mask_data)

        return confidence, state_feat


    def get_label_value(self, env_change_detected, grasp_succeeded, grasp_effective, target_grasped, target_oriented,
                        next_depth_heightmap, next_target_mask_heightmap, next_grasp_mask_heightmap):
        # Compute current reward
        current_reward = 0.0
        # if grasp_succeeded:
        #     current_reward = 0.4
        if grasp_effective:
            current_reward = 1.0
        if target_oriented:
            current_reward = 1.0
        if target_grasped:
            current_reward = 2.0

        # print('%%%current_reward', current_reward)

        # Compute future reward
        # if not env_change_detected:
        #     future_reward = 0
        # else:
            ###################
            # Generate grasp candidates
            # point_cloud = get_pointcloud(heightmap_resolution, valid_depth_heightmap, workspace_limits)
            # grasps, grasp_mask_heightmaps, num_grasps = grasps_generator(target_center, point_cloud)
            #
            # sampled_inds = np.random.choice(np.arange(num_grasps), 100, replace=False)
            # confs, grasp_inds, rot_inds = [], [], []
            # for i in sampled_inds:
            #     grasp_mask_heightmap = grasp_mask_heightmaps[i][0]
            #     confidence, _ = self.forward(depth_heightmap, target_mask_heightmap, grasp_mask_heightmap)
            #     confs.append(confidence)
            #     grasp_inds.append(grasp_mask_heightmaps[i][1])
            #     rot_inds.append(grasp_mask_heightmaps[i][2])
            #
            # grasp_inds = np.hstack((np.array(rot_inds).reshape((-1, 1)), np.array(grasp_inds)))
            #
            # best_conf = np.max(confs)
            # best_ind = np.argmax(confs)
            # best_grasp_ind = grasp_inds[best_ind]
            ###################
            # next_conf, _ = self.forward(
            #     next_depth_heightmap, next_target_mask_heightmap, next_grasp_mask_heightmap)
            # future_reward = 1 # ***

        expected_reward = current_reward #+ self.future_reward_discount * future_reward

        # print('Expected reward: %f + %f x %f = %f' % (
        # # current_reward, self.future_reward_discount, future_reward, expected_reward))
        # current_reward, self.future_reward_discount, expected_reward))

        return expected_reward, current_reward


    # Compute labels and backpropagate
    def backprop(self, depth_heightmap, target_mask_heightmap, grasp_mask_heightmap, label_value):

        # Compute loss and backward pass
        self.optimizer.zero_grad()

        # Do forward pass
        conf, _ = self.forward(depth_heightmap, target_mask_heightmap, grasp_mask_heightmap)

        if self.use_cuda:
            loss = self.criterion(conf, torch.Tensor([label_value]).float().squeeze().cuda())
        else:
            loss = self.criterion(conf, torch.Tensor([label_value]).float().squeeze())
        print('output: ', conf.item())
        loss.backward()
        loss_value = loss.cpu().detach().numpy()
        print('loss_value', loss_value)

        self.optimizer.step()

        return loss_value

