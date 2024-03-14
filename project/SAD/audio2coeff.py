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
import os 
import torch
import torch.nn as nn

from SAD.audio2pose import Audio2Pose
from SAD.audio2exp import Audio2Exp
from typing import Dict

import todos
import pdb


class Audio2Coeff(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio2exp_model = Audio2Exp()
        self.audio2pose_model = Audio2Pose()

    def forward(self, batch: Dict[str, torch.Tensor], pose_style:int=0):
        # batch is dict:
        #     tensor [audio_mels] size: [1, 200, 1, 80, 16], min: -4.0, max: 2.590095, mean: -1.017794
        #     tensor [image_exp_pose] size: [1, 200, 70], min: -1.156697, max: 1.459776, mean: 0.023419
        #     tensor [audio_ratio] size: [1, 200, 1], min: 0.0, max: 1.0, mean: 0.6575
        # [pose_style] value: '0'

        with torch.no_grad():
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            exp_pred= self.audio2exp_model(batch) # Audio2Exp(...)
            # tensor [exp_pred] size: [1, 200, 64], min: -1.673844, max: 1.242088, mean: -0.011497

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            #46 styles can be selected 
            batch['class'] = torch.LongTensor([pose_style]).to(exp_pred.device)

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            pose_pred = self.audio2pose_model(batch) # Audio2Pose(...)
            # tensor [pose_pred] size: [1, 200, 6], min: -0.809749, max: 0.27462, mean: -0.102681
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            # coeffs_pred = torch.cat((exp_pred, pose_pred), dim=-1)
            coeffs_pred = torch.cat((exp_pred, pose_pred), dim=2)
            # tensor [coeffs_pred] size: [1, 200, 70], min: -1.690973, max: 1.272287, mean: -0.021126

            return coeffs_pred # coeffs_pred.size() -- [1, 200, 70]


if __name__ == "__main__":
    model = Audio2Coeff()
    model = torch.jit.script(model)    
    print(model)
    # ==> OK