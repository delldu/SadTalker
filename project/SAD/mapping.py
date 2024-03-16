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
from SAD.util import keypoint_transform

# from typing import Tuple
import todos

import pdb

class MappingNet(nn.Module):
    '''
    src/config/facerender.yaml

      mapping_params:
          coeff_nc: 70
          descriptor_nc: 1024
          layer: 3
          num_kp: 15
          num_bins: 66

    '''
    def __init__(self, 
        coeff_nc=70, 
        descriptor_nc=1024, 
        layer=3, 
        num_kp=15, 
        num_bins=66,
    ):
        super().__init__()

        self.layer = layer
        self.first = nn.Sequential(
            nn.Conv1d(coeff_nc, descriptor_nc, kernel_size=7, padding=0, bias=True))

        for i in range(layer):
            net = nn.Sequential(nn.LeakyReLU(0.1),
                nn.Conv1d(descriptor_nc, descriptor_nc, kernel_size=3, padding=0, dilation=3))
            setattr(self, 'encoder' + str(i), net)   

        self.pooling = nn.AdaptiveAvgPool1d(1)

        self.fc_roll = nn.Linear(descriptor_nc, num_bins)
        self.fc_pitch = nn.Linear(descriptor_nc, num_bins)
        self.fc_yaw = nn.Linear(descriptor_nc, num_bins)
        self.fc_t = nn.Linear(descriptor_nc, 3)
        self.fc_exp = nn.Linear(descriptor_nc, 3*num_kp)

    def forward(self, canonical_kp, input_3dmm):
        """Audio Encoder --> Mapping ?"""
        # tensor [input_3dmm] size: [70, 27], min: -1.156697, max: 1.459776, mean: 0.023419

        out = self.first(input_3dmm.unsqueeze(0))
        # tensor [out] size: [1, 1024, 21], min: -1.797052, max: 0.736073, mean: -0.228992

        # to support torch.jit.script
        # for i in range(self.layer): # self.layer -- 3
        #     model = getattr(self, 'encoder' + str(i))
        #     out = model(out) + out[:,:, 3:-3]
        # out = self.encoder0(out) + out[:,:, 3:-3] # out[:,:, 3:-3].size() -- [1, 1024, 15]
        # out = self.encoder1(out) + out[:,:, 3:-3] # out[:,:, 3:-3].size() -- [1, 1024, 9]
        # out = self.encoder2(out) + out[:,:, 3:-3] # out[:,:, 3:-3].size() -- [1, 1024, 3]

        out = self.encoder0(out) + out[:,:, 3:18] # out[:,:, 3:-3].size() -- [1, 1024, 15]
        out = self.encoder1(out) + out[:,:, 3:12] # out[:,:, 3:-3].size() -- [1, 1024, 9]
        out = self.encoder2(out) + out[:,:, 3:6] # out[:,:, 3:-3].size() -- [1, 1024, 3]
        # out.size() -- [1, 1024, 3]

        out = self.pooling(out)
        out = out.view(out.shape[0], -1)

        yaw = self.fc_yaw(out)
        pitch = self.fc_pitch(out)
        roll = self.fc_roll(out)
        trans = self.fc_t(out)
        exp = self.fc_exp(out)

        # tensor [yaw] size: [1, 66], min: -3.959461, max: 4.397331, mean: 0.079592
        # tensor [pitch] size: [1, 66], min: -4.535993, max: 5.854048, mean: -0.505516
        # tensor [roll] size: [1, 66], min: -3.812059, max: 6.655958, mean: -0.263339
        # tensor [trans] size: [1, 3], min: -0.058004, max: 0.227935, mean: 0.068895
        # tensor [exp] size: [1, 45], min: -0.102243, max: 0.015078, mean: -0.002095

        return keypoint_transform(canonical_kp, yaw, pitch, roll, trans, exp)