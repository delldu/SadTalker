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

from SAD.audio2pose import Audio2Pose
from SAD.audio2exp import Audio2Exp

import todos
import pdb


class Audio2Coeff(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio2exp_model = Audio2Exp()
        self.audio2pose_model = Audio2Pose()

    def forward(self, audio_mels, audio_ratio, image_exp_pose):
        pose_style:int = 0 # xxxx_8888
        # tensor [audio_mels] size: [1, 200, 1, 80, 16], min: -4.0, max: 2.590095, mean: -1.017794
        # tensor [audio_ratio] size: [1, 200, 1], min: 0.0, max: 1.0, mean: 0.6575
        # tensor [image_exp_pose] size: [1, 200, 70], min: -1.156697, max: 1.459776, mean: 0.023419

        with torch.no_grad():
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            exp_pred= self.audio2exp_model(audio_mels, image_exp_pose, audio_ratio) # Audio2Exp(...)
            # tensor [exp_pred] size: [1, 200, 64], min: -1.673844, max: 1.242088, mean: -0.011497

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            # 46 styles can be selected 
            class_id = torch.LongTensor([pose_style]).to(exp_pred.device)

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            pose_pred = self.audio2pose_model(audio_mels, image_exp_pose, audio_ratio, class_id) # Audio2Pose(...)
            # tensor [pose_pred] size: [1, 200, 6], min: -0.809749, max: 0.27462, mean: -0.102681
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            coeffs_pred = torch.cat((exp_pred, pose_pred), dim=2)
            # tensor [coeffs_pred] size: [1, 200, 70], min: -1.690973, max: 1.272287, mean: -0.021126

            return coeffs_pred.squeeze(0) # coeffs_pred.size() -- [200, 70]
