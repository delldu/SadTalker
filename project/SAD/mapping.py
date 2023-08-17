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
from SAD.util import load_weights

from typing import Dict
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
        super(MappingNet, self).__init__()

        self.layer = layer
        self.first = nn.Sequential(
            torch.nn.Conv1d(coeff_nc, descriptor_nc, kernel_size=7, padding=0, bias=True))

        for i in range(layer):
            net = nn.Sequential(nn.LeakyReLU(0.1),
                torch.nn.Conv1d(descriptor_nc, descriptor_nc, kernel_size=3, padding=0, dilation=3))
            setattr(self, 'encoder' + str(i), net)   

        self.pooling = nn.AdaptiveAvgPool1d(1)

        self.fc_roll = nn.Linear(descriptor_nc, num_bins)
        self.fc_pitch = nn.Linear(descriptor_nc, num_bins)
        self.fc_yaw = nn.Linear(descriptor_nc, num_bins)
        self.fc_t = nn.Linear(descriptor_nc, 3)
        self.fc_exp = nn.Linear(descriptor_nc, 3*num_kp)

        # load_weights(self, "models/MappingNet.pth")

    def forward(self, input_3dmm) -> Dict[str, torch.Tensor]:
        out = self.first(input_3dmm)

        # to support torch.jit.script
        # for i in range(self.layer): # self.layer -- 3
        #     model = getattr(self, 'encoder' + str(i))
        #     out = model(out) + out[:,:, 3:-3]
        out = self.encoder0(out) + out[:,:, 3:-3]
        out = self.encoder1(out) + out[:,:, 3:-3]
        out = self.encoder2(out) + out[:,:, 3:-3]

        out = self.pooling(out)
        out = out.view(out.shape[0], -1)

        yaw = self.fc_yaw(out)
        pitch = self.fc_pitch(out)
        roll = self.fc_roll(out)
        t = self.fc_t(out)
        exp = self.fc_exp(out)

        return {'yaw': yaw, 'pitch': pitch, 'roll': roll, 't': t, 'exp': exp} 


if __name__ == "__main__":
    model = MappingNet()
    model = torch.jit.script(model)    
    print(model)
    # ==> OK

