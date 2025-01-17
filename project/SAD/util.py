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
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import remove_spectral_norm
import torchvision.transforms as T
from PIL import ImageDraw

from typing import Tuple

import todos
import pdb

def remove_sadkernel_spectral_norm(model):
    remove_spectral_norm(model.sadkernel_model.decoder.G_middle_0.conv_0)
    remove_spectral_norm(model.sadkernel_model.decoder.G_middle_0.conv_1)
    remove_spectral_norm(model.sadkernel_model.decoder.G_middle_1.conv_0)
    remove_spectral_norm(model.sadkernel_model.decoder.G_middle_1.conv_1)
    remove_spectral_norm(model.sadkernel_model.decoder.G_middle_2.conv_0)
    remove_spectral_norm(model.sadkernel_model.decoder.G_middle_2.conv_1)
    remove_spectral_norm(model.sadkernel_model.decoder.G_middle_3.conv_0)
    remove_spectral_norm(model.sadkernel_model.decoder.G_middle_3.conv_1)
    remove_spectral_norm(model.sadkernel_model.decoder.G_middle_4.conv_0)
    remove_spectral_norm(model.sadkernel_model.decoder.G_middle_4.conv_1)
    remove_spectral_norm(model.sadkernel_model.decoder.G_middle_5.conv_0)
    remove_spectral_norm(model.sadkernel_model.decoder.G_middle_5.conv_1)

    remove_spectral_norm(model.sadkernel_model.decoder.up_0.conv_0)
    remove_spectral_norm(model.sadkernel_model.decoder.up_0.conv_1)
    remove_spectral_norm(model.sadkernel_model.decoder.up_0.conv_s)
    remove_spectral_norm(model.sadkernel_model.decoder.up_1.conv_0)
    remove_spectral_norm(model.sadkernel_model.decoder.up_1.conv_1)
    remove_spectral_norm(model.sadkernel_model.decoder.up_1.conv_s)   

def load_weights(model, model_path):
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    if os.path.exists(checkpoint):
        print(f"Loading model weight from {checkpoint} ...")
        model.load_state_dict(torch.load(checkpoint))
        remove_sadkernel_spectral_norm(model)
    else:
        print("-" * 32, "Warnning", "-" * 32)
        print(f"Model weight file '{checkpoint}' not exist !!!")

def make_coordinate_grid(spatial_size: Tuple[int, int, int]):
    d, h, w = spatial_size
    x = torch.arange(w)
    y = torch.arange(h)
    z = torch.arange(d)

    x = 2.0 * (x / (w - 1.0)) - 1.0
    y = 2.0 * (y / (h - 1.0)) - 1.0
    z = 2.0 * (z / (d - 1.0)) - 1.0

    yy = y.view(1, -1, 1).repeat(d, 1, w)
    xx = x.view(1, 1, -1).repeat(d, h, 1)
    zz = z.view(-1, 1, 1).repeat(1, h, w)

    meshed = torch.cat([xx.unsqueeze_(3), yy.unsqueeze_(3), zz.unsqueeze_(3)], 3)

    return meshed


def headpose_pred_to_degree(pred):
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).type_as(pred).to(pred.device)
    pred = F.softmax(pred, dim=1)
    degree = torch.sum(pred*idx_tensor, 1) * 3.0 - 99.0
    return degree

def get_rotation_matrix(yaw, pitch, roll):
    # tensor [yaw] size: [1], min: 14.669785, max: 14.669785, mean: 14.669785
    # tensor [pitch] size: [1], min: -14.523376, max: -14.523376, mean: -14.523376
    # tensor [roll] size: [1], min: -3.038132, max: -3.038132, mean: -3.038132

    yaw = yaw / 180 * 3.14
    pitch = pitch / 180 * 3.14
    roll = roll / 180 * 3.14

    roll = roll.unsqueeze(1)
    pitch = pitch.unsqueeze(1)
    yaw = yaw.unsqueeze(1)

    pitch_mat = torch.cat([torch.ones_like(pitch), torch.zeros_like(pitch), torch.zeros_like(pitch), 
                          torch.zeros_like(pitch), torch.cos(pitch), -torch.sin(pitch),
                          torch.zeros_like(pitch), torch.sin(pitch), torch.cos(pitch)], dim=1)
    pitch_mat = pitch_mat.view(pitch_mat.shape[0], 3, 3)
    # (Pdb) pitch_mat
    #         [[ 1.0000,  0.0000,  0.0000],
    #          [ 0.0000,  0.9931,  0.1170],
    #          [ 0.0000, -0.1170,  0.9931]]], device='cuda:0')

    yaw_mat = torch.cat([torch.cos(yaw), torch.zeros_like(yaw), torch.sin(yaw), 
                           torch.zeros_like(yaw), torch.ones_like(yaw), torch.zeros_like(yaw),
                           -torch.sin(yaw), torch.zeros_like(yaw), torch.cos(yaw)], dim=1)
    yaw_mat = yaw_mat.view(yaw_mat.shape[0], 3, 3)
    # (Pdb) yaw_mat
    # tensor([[[ 1.0000,  0.0000, -0.0021],
    #          [ 0.0000,  1.0000,  0.0000],
    #          [ 0.0021,  0.0000,  1.0000]], device='cuda:0'

    roll_mat = torch.cat([torch.cos(roll), -torch.sin(roll), torch.zeros_like(roll),  
                         torch.sin(roll), torch.cos(roll), torch.zeros_like(roll),
                         torch.zeros_like(roll), torch.zeros_like(roll), torch.ones_like(roll)], dim=1)
    roll_mat = roll_mat.view(roll_mat.shape[0], 3, 3)
    # (Pdb) roll_mat
    # tensor([[[ 1.0000, -0.0071,  0.0000],
    #          [ 0.0071,  1.0000,  0.0000],
    #          [ 0.0000,  0.0000,  1.0000]], device='cuda:0')

    # tensor [pitch_mat] size: [1, 3, 3], min: -0.250651, max: 1.0, mean: 0.326239
    # tensor [yaw_mat] size: [1, 3, 3], min: -0.253122, max: 1.0, mean: 0.326097
    # tensor [roll_mat] size: [1, 3, 3], min: -0.052974, max: 1.0, mean: 0.333021

    rot_mat = torch.einsum('bij,bjk,bkm->bim', pitch_mat, yaw_mat, roll_mat)
    # (Pdb) rot_mat
    # tensor([[[ 1.0000, -0.0071, -0.0021],
    #          [ 0.0072,  0.9931,  0.1170],
    #          [ 0.0012, -0.1170,  0.9931]], device='cuda:0'
    return rot_mat # size() -- [1, 3, 3]

def keypoint_transform(kp, yaw, pitch, roll, trans, express):
    # kp -- [2, 15, 3]
    # yaw.size() --[2, 66]

    yaw = headpose_pred_to_degree(yaw) 
    pitch = headpose_pred_to_degree(pitch)
    roll = headpose_pred_to_degree(roll)
    rot_mat = get_rotation_matrix(yaw, pitch, roll)    # (bs, 3, 3)

    # keypoint rotation
    kp_rotated = torch.einsum('bmp,bkp->bkm', rot_mat, kp)

    # Keypoint translation
    # (Pdb) trans
    # tensor([[ 0.0084, -0.0088,  0.2165],
    #         [ 0.0084, -0.0088,  0.2165]], device='cuda:0')
    trans[:, 0] = trans[:, 0]*0.0
    trans[:, 2] = trans[:, 2]*0.0
    trans = trans.unsqueeze(1).repeat(1, kp.shape[1], 1)
    kp_t = kp_rotated + trans

    # Add expression deviation 
    express = express.view(express.shape[0], -1, 3) # [1, 45] ==> [1, 15, 3]
    kp_transformed = kp_t + express

    return kp_transformed


class DownBlock2d(nn.Module):
    """Downsampling block for use in encoder. """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, 
                        kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = nn.BatchNorm2d(out_features) # onnx ???
        self.pool = nn.AvgPool2d(kernel_size=(2, 2)) # onnx ???

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out

class UpBlock3d(nn.Module):
    """Upsampling block for use in decoder. """
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, 
                        kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = nn.BatchNorm3d(out_features)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=(1.0, 2.0, 2.0))
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out

def draw_keypoint(t, kp):
    kp = (kp + 1.0)/2.0
    kp = kp.clamp(0.0, 1.0)
    kp = kp * 512.0

    image = T.functional.to_pil_image(t[0, :, :, :])
    draw = ImageDraw.Draw(image)

    for i in range(kp.size(1)):
        x, y, z = kp[0, i]
        x = x.item()
        y = y.item()
        # box = (x1, y1, x2, y2)
        box = (int(x) - 2, int(y) - 2, int(x) + 2, int(y) + 2)
        draw.ellipse(box, fill=None, outline="#FF0000", width=1)

    t[0, :, :, :] = T.functional.to_tensor(image)
    return t

