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
# '''
#  * SAD inference module
# '''

# import os
import pdb
# import numpy as np
import torch
from torch import nn
# import torch.nn.functional as F
# import torchvision.transforms as T

from SAD.image2coeff import Image2Coeff
from SAD.audio2coeff import Audio2Coeff
from SAD.sadkernel import SADKernel
from SAD.keypoint_detector import KPDetector
from SAD.mapping import MappingNet
from SAD.util import load_weights

class SADModel(nn.Module):
    def __init__(self):
        r""" 
        SadTalker: Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven 
            Single Image Talking Face Animation
        """
        super().__init__()
        self.image2coffe_model = Image2Coeff()
        self.audio2coffe_model = Audio2Coeff()
        self.sadkernel_model = SADKernel()
        self.kpdetector_model = KPDetector()
        self.mappingnet_model = MappingNet()

        load_weights(self, model_path="models/SAD.pth")

    def forward(self, audio, image):
        # image to Bx3x512x512
        return image

