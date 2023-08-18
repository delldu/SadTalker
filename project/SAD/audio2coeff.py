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
from SAD.debug import debug_var

from typing import Dict

import pdb


class Audio2Coeff(nn.Module):
    def __init__(self):
        super(Audio2Coeff, self).__init__()
        self.audio2exp_model = Audio2Exp()
        self.audio2pose_model = Audio2Pose()

    def forward(self, batch: Dict[str, torch.Tensor], pose_style:int=0):
        # batch is dict:
        #     tensor [audio_mels] size: [1, 200, 1, 80, 16] , min: tensor(-4., device='cuda:0') , max: tensor(2.5901, device='cuda:0')
        #     tensor [image_exp_pose] size: [1, 200, 70] , min: tensor(-1.1567, device='cuda:0') , max: tensor(1.4598, device='cuda:0')
        ##    [num_frames] value: 200
        #     tensor [audio_ratio] size: [1, 200, 1] , min: tensor(0., device='cuda:0') , max: tensor(1., device='cuda:0')
        #     tensor [class] size: [1] , min: tensor(0, device='cuda:0') , max: tensor(0, device='cuda:0')

        with torch.no_grad():
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            exp_pred= self.audio2exp_model(batch) # Audio2Exp(...)
            # exp_pred.size() -- [1, 200, 64]
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            #46 styles can be selected 
            batch['class'] = torch.LongTensor([pose_style]).to(exp_pred.device)

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            pose_pred = self.audio2pose_model(batch) # Audio2Pose(...)
            # pose_pred.size() -- [1, 200, 6]
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            # exp_pred.size() -- [1, 200, 64]
            # pose_pred.size() -- [1, 200, 6]
            coeffs_pred = torch.cat((exp_pred, pose_pred), dim=-1)

            return coeffs_pred # coeffs_pred.size() -- [1, 200, 70]


if __name__ == "__main__":
    model = Audio2Coeff()
    model = torch.jit.script(model)    
    print(model)
    # ==> OK