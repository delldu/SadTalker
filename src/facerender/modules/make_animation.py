import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm 
from typing import Dict
import pdb

def headpose_pred_to_degree(pred):
    device = pred.device
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).type_as(pred).to(device)
    pred = F.softmax(pred, dim=1)
    degree = torch.sum(pred*idx_tensor, 1) * 3 - 99
    return degree

def get_rotation_matrix(yaw, pitch, roll):
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
    # tensor([[[ 1.0000,  0.0000,  0.0000],
    #          [ 0.0000,  0.9931,  0.1170],
    #          [ 0.0000, -0.1170,  0.9931]],

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
    #          [ 0.0021,  0.0000,  1.0000]],

    #         [[ 1.0000,  0.0000, -0.0021],
    #          [ 0.0000,  1.0000,  0.0000],
    #          [ 0.0021,  0.0000,  1.0000]]], device='cuda:0')

    roll_mat = torch.cat([torch.cos(roll), -torch.sin(roll), torch.zeros_like(roll),  
                         torch.sin(roll), torch.cos(roll), torch.zeros_like(roll),
                         torch.zeros_like(roll), torch.zeros_like(roll), torch.ones_like(roll)], dim=1)
    roll_mat = roll_mat.view(roll_mat.shape[0], 3, 3)
    # (Pdb) roll_mat
    # tensor([[[ 1.0000, -0.0071,  0.0000],
    #          [ 0.0071,  1.0000,  0.0000],
    #          [ 0.0000,  0.0000,  1.0000]],

    #         [[ 1.0000, -0.0071,  0.0000],
    #          [ 0.0071,  1.0000,  0.0000],
    #          [ 0.0000,  0.0000,  1.0000]]], device='cuda:0')

    rot_mat = torch.einsum('bij,bjk,bkm->bim', pitch_mat, yaw_mat, roll_mat)
    # (Pdb) rot_mat
    # tensor([[[ 1.0000, -0.0071, -0.0021],
    #          [ 0.0072,  0.9931,  0.1170],
    #          [ 0.0012, -0.1170,  0.9931]],

    #         [[ 1.0000, -0.0071, -0.0021],
    #          [ 0.0072,  0.9931,  0.1170],
    #          [ 0.0012, -0.1170,  0.9931]]], device='cuda:0')

    return rot_mat

def keypoint_transformation(canonical_kp: Dict[str, torch.Tensor], he: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # canonical_kp['value'].size() -- [2, 15, 3]
    # he.keys() -- ['yaw', 'pitch', 'roll', 't', 'exp']

    kp = canonical_kp['value']    # (bs, k, 3) 
    yaw, pitch, roll= he['yaw'], he['pitch'], he['roll']
    # he['yaw'].size() --[2, 66]

    yaw = headpose_pred_to_degree(yaw) 
    pitch = headpose_pred_to_degree(pitch)
    roll = headpose_pred_to_degree(roll)

    if 'yaw_in' in he: # False
        yaw = he['yaw_in']
    if 'pitch_in' in he: # False
        pitch = he['pitch_in']
    if 'roll_in' in he: # False
        roll = he['roll_in']

    rot_mat = get_rotation_matrix(yaw, pitch, roll)    # (bs, 3, 3)

    t, exp = he['t'], he['exp']
    # keypoint rotation
    kp_rotated = torch.einsum('bmp,bkp->bkm', rot_mat, kp)

    # keypoint translation
    # (Pdb) he['t']
    # tensor([[ 0.0084, -0.0088,  0.2165],
    #         [ 0.0084, -0.0088,  0.2165]], device='cuda:0')
    t[:, 0] = t[:, 0]*0
    t[:, 2] = t[:, 2]*0
    t = t.unsqueeze(1).repeat(1, kp.shape[1], 1)
    kp_t = kp_rotated + t

    # add expression deviation 
    exp = exp.view(exp.shape[0], -1, 3) # [2, 45] ==> [2, 15, 3]
    kp_transformed = kp_t + exp

    return {'value': kp_transformed}

# xxxx8888 **************************************************************
def make_animation(source_image, image_semantics, audio_semantics, generator, kp_detector, mapping):
    with torch.no_grad():
        predictions = []

        # source_image.size() -- [2, 3, 256, 256]
        # image_semantics.size() -- [2, 70, 27]
        # audio_semantics.size() -- [2, 100, 70, 27]

        # kp_detector -- KPDetector(...)
        canonical_kp = kp_detector(source_image) # canonical_kp['value'].size() -- [2, 15, 3]

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # mapping -- MappingNet(...)            
        he_source = mapping(image_semantics) # head estimation rotation matrix ?
        # he_source.keys() -- ['yaw', 'pitch', 'roll', 't', 'exp']
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!            
        # xxxx7777 Step 3
        image_kp = keypoint_transformation(canonical_kp, he_source) 

        # xxxx9999 Step 4
        for frame_idx in tqdm(range(audio_semantics.shape[1]), 'Face Renderer:'):
            # still check the dimension
            audio_semantics_frame = audio_semantics[:, frame_idx]
            he_driving = mapping(audio_semantics_frame)
            audio_kp = keypoint_transformation(canonical_kp, he_driving)

            # generator -- SADKernel(...)
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # source_image.size() -- [2, 3, 256, 256]
            # image_kp['value'].size() -- [2, 15, 3]
            # audio_kp['value'].size() -- [2, 15, 3]

            # generator -- SADKernel(...)
            out = generator(source_image, audio_kp=audio_kp, image_kp=image_kp)
            
            # out.keys() -- ['mask', 'occlusion_map', 'prediction']
            # out['prediction'].size() -- [2, 3, 256, 256]
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            predictions.append(out['prediction'])
        predictions_ts = torch.stack(predictions, dim=1)

    return predictions_ts

