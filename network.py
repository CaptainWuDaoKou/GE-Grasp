import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def pool2x2(x):
    return nn.MaxPool2d(kernel_size=2, stride=2)(x)


def upsample2(x):
    return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class FeatureTunk(nn.Module):

    def __init__(self, pretrained=False):
        super(FeatureTunk, self).__init__()

        # self.color_extractor = BasicBlock(3, 3)
        self.color_extractor = BasicBlock(1, 1) # grasp extractor
        self.depth_extractor = BasicBlock(1, 1)
        self.mask_extractor = BasicBlock(1, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.dense121 = torchvision.models.densenet.densenet121(pretrained=pretrained).features
        self.dense121.conv0 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, color, depth, mask):

        return self.dense121(torch.cat((self.color_extractor(color), self.depth_extractor(depth), self.mask_extractor(mask)), dim=1))
