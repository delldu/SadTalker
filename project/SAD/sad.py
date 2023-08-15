# '''
#  * SAD inference module
# '''
import os
import pdb
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T


class SADModel(nn.Module):
    def __init__(self):
        r""" 
        SadTalker: Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven 
            Single Image Talking Face Animation
        """
        super().__init__()
        self.model = nn.Identity()

        self.load_weights()

    def forward(self, wav, image):
        # image to 512x512
        return self.model(image)

    def load_weights(self, model_path="models/SAD.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path

        if os.path.exists(checkpoint):
            print(f"Loading model weight from {checkpoint} ...")
            self.load_state_dict(torch.load(checkpoint))
        else:
            print("-" * 32, "Warnning", "-" * 32)
            print(f"model weight file '{checkpoint}'' not exist !!!")
