#!/usr/bin/env python

from collections import OrderedDict
# import torch
import torch.nn as nn
from torch.autograd import Variable
from network import FeatureTunk


class evaluator_net(nn.Module):

    def __init__(self, use_cuda):
        super(evaluator_net, self).__init__()
        self.use_cuda = use_cuda

        # Initialize network trunks with DenseNet pre-trained on ImageNet
        self.feature_tunk = FeatureTunk()

        self.num_rotations = 16

        # Construct network branches for pushing and grasping
        self.graspnet = nn.Sequential(OrderedDict([
            ('grasp-norm0', nn.BatchNorm2d(1024)),
            ('grasp-relu0', nn.ReLU(inplace=True)),
            ('grasp-conv0', nn.Conv2d(1024, 128, kernel_size=3, stride=1, padding=1, bias=True)),
            ('grasp-maxpool0', nn.MaxPool2d(20)),
        ]))
        self.fc1 = nn.Linear(128, 1)# regression

        # Initialize network weights
        for m in self.named_modules():
            if 'grasp-' in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    nn.init.kaiming_normal_(m[1].weight.data)
                elif isinstance(m[1], nn.BatchNorm2d):
                    m[1].weight.data.fill_(1)
                    m[1].bias.data.zero_()
        self.confidence = 0


    def forward(self, input_depth_data, input_target_mask_data, input_grasp_mask_data):

        if self.use_cuda:
            depth = Variable(input_depth_data).cuda()
            target_mask = Variable(input_target_mask_data).cuda()
            grasp_mask = Variable(input_grasp_mask_data).cuda()
        else:
            depth = Variable(input_depth_data)
            target_mask = Variable(input_target_mask_data)
            grasp_mask = Variable(input_grasp_mask_data)

        # Compute intermediate features
        interm_feat = self.feature_tunk(depth, target_mask, grasp_mask)

        # Forward pass
        conf = self.fc1(self.graspnet(interm_feat).squeeze()).squeeze() # regression

        return conf, interm_feat
