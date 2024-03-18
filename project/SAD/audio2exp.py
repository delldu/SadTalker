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
from torch import nn

from typing import List
import todos
import pdb

class Audio2Exp(nn.Module):
    def __init__(self):
        super().__init__()
        self.netG = Audio2ExpWrapperV2()

    def forward(self, audio_mels, image_exp_pose, audio_ratio):
        # tensor [audio_mels] size: [1, 200, 80, 16], min: -4.0, max: 2.590095, mean: -1.017794
        # tensor [image_exp_pose] size: [1, 200, 70], min: -1.156697, max: 1.459776, mean: 0.023419
        # tensor [audio_ratio] size: [1, 200, 1], min: 0.0, max: 1.0, mean: 0.58

        T = audio_mels.shape[1] # ==> 200
        # exp_predict_list: List[torch.Tensor] = []
        exp_predict_list = []

        for i in range(0, T, 10): # every 10 frames
            audio_mel = audio_mels[:, i:i+10, :, :].view(-1, 1, 80, 16) # size() -- [10, 1, 80, 16]

            image_exp = image_exp_pose[:, i:i+10, 0:64] # size() -- [1, 10, 64]
            audio_ratio2 = audio_ratio[:, i:i+10, :]  # size() -- [1, 10, 1]
            
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # self.netG -- Audio2ExpWrapperV2(...)
            y  = self.netG(audio_mel, image_exp, audio_ratio2) # [1, 200, 64]
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!            

            exp_predict_list += [y] # size() -- [1, 10, 64]

        # exp_predict_list is list: len = 20
        #     tensor [item] size: [1, 10, 64], min: -1.311815, max: 1.241706, mean: -0.000448
        #     ...
        #     tensor [item] size: [1, 10, 64], min: -1.622588, max: 1.211561, mean: 0.006403
        return torch.cat(exp_predict_list, dim=1) # size() -- [1, 200, 64]


class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding):
        super().__init__()
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                        )
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)


class Conv2dWithRes(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding):
        super().__init__()
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                        )
        self.act = nn.ReLU()
        # self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        out += x
        return self.act(out)

class Audio2ExpWrapperV2(nn.Module):
    ''' come from wav2lip '''
    def __init__(self):
        super().__init__()
        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2dWithRes(32, 32, kernel_size=3, stride=1, padding=1),
            Conv2dWithRes(32, 32, kernel_size=3, stride=1, padding=1),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2dWithRes(64, 64, kernel_size=3, stride=1, padding=1),
            Conv2dWithRes(64, 64, kernel_size=3, stride=1, padding=1),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2dWithRes(128, 128, kernel_size=3, stride=1, padding=1),
            Conv2dWithRes(128, 128, kernel_size=3, stride=1, padding=1),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2dWithRes(256, 256, kernel_size=3, stride=1, padding=1),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
        )
        self.mapping1 = nn.Linear(512+64+1, 64)

    def forward(self, audio_mel, image_exp, audio_ratio):
        # tensor [audio_mel] size: [10, 1, 80, 16], min: -3.269791, max: 0.703544, mean: -1.632042
        # tensor [image_exp] size: [1, 10, 64], min: -1.156697, max: 1.459776, mean: 0.036036
        # tensor [audio_ratio] size: [1, 10, 1], min: 0.0, max: 0.0, mean: 0.0
        audio_mel = self.audio_encoder(audio_mel).view(audio_mel.size(0), -1) # size() -- [10, 512]
        ref_reshape = image_exp.reshape(audio_mel.size(0), -1) # size() -- [10, 64]
        audio_ratio = audio_ratio.reshape(audio_mel.size(0), -1) # size() -- [10, 1]
        
        y = self.mapping1(torch.cat([audio_mel, ref_reshape, audio_ratio], dim=1)) # size() -- [10, 64]
        out = y.reshape(image_exp.shape[0], image_exp.shape[1], -1) # size() -- [1, 10, 64]
        # tensor [out] size: [1, 10, 64], min: -1.311815, max: 1.241706, mean: -0.000448

        return out
