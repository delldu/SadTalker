# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2023(18588220928@163.com) All Rights Reserved.
# ***
# ***    File Author: Dell, Wed 16 Aug 2023 04:17:55 PM CST
# ***
# ************************************************************************************/
#

import torch
import torch.nn as nn
import torch.nn.functional as F

import todos
import pdb


class Image2Coeff(nn.Module):
    fc_dim=257
    def __init__(self):
        super().__init__()

        self.backbone = ResNet([3, 4, 6, 3], {'num_classes': self.fc_dim})
        last_dim = 2048
        self.final_layers = nn.ModuleList([
            conv1x1(last_dim, 80, bias=True), # id layer
            conv1x1(last_dim, 64, bias=True), # exp layer
            conv1x1(last_dim, 80, bias=True), # tex layer
            conv1x1(last_dim, 3, bias=True),  # angle layer
            conv1x1(last_dim, 27, bias=True), # gamma layer
            conv1x1(last_dim, 2, bias=True),  # tx, ty
            conv1x1(last_dim, 1, bias=True)   # tz
        ])


    def forward(self, x):
        # tensor [x] size: [1, 3, 512, 512], min: 0.117647, max: 1.0, mean: 0.644081

        # resize to Bx3x224x224
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        x = self.backbone(x) # size() -- [1, 2048, 1, 1]
        output = []
        for layer in self.final_layers:
            output.append(layer(x))
        x = torch.flatten(torch.cat(output, dim=1), 1) # x.size() -- [1, 257]

        ##################################################
        # id = x[:, :80]
        # exp = x[:, 80: 144], dim = 64
        # tex = x[:, 144: 224]
        # angle = x[:, 224: 227], dim = 3
        # gamma = x[:, 227: 254]
        # trans = x[:, 254: 257], dim = 3
        # ==> exp + angle + trans, total dim is 70
        ##################################################

        output = torch.cat((x[:, 80:144], x[:, 224:227], x[:, 254:257]), dim=1)
        # tensor [output] size: [1, 70], min: -1.156697, max: 1.459776, mean: 0.023419

        return output


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, bias: bool = False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        groups = 1,
        base_width = 64,
        dilation = 1,
        norm_layer = nn.BatchNorm2d,
    ):
        super().__init__()

        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # self.downsample === None
        # if self.downsample is not None: # True or False
        #     identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class BottleneckWithDownsample(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample= None,
        groups = 1,
        base_width = 64,
        dilation = 1,
        norm_layer = nn.BatchNorm2d,
    ):
        super().__init__()

        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()

        assert downsample is not None, "downsample must be not None"
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self,
        layers = [3, 4, 6, 3],
        num_classes = 1000,
        groups = 1,
        width_per_group = 64,
        norm_layer = nn.BatchNorm2d,
    ):
        super().__init__()
        # num_classes = 257

        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


    def _make_layer(self, planes, blocks, stride = 1):
        expansion = Bottleneck.expansion # ---- 4
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * expansion, stride),
                norm_layer(planes * expansion),
            )

        layers = []
        if downsample is not None:
            layers.append(BottleneckWithDownsample(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, self.dilation, norm_layer))
        else:
            layers.append(Bottleneck(self.inplanes, planes, stride, self.groups, 
                            self.base_width, self.dilation, norm_layer))

        self.inplanes = planes * expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, groups=self.groups,
                            base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return x
