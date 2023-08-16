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
from SAD.util import load_weights, keypoint_transform
from SAD.audio import (
        melspectrogram, 
        get_blink_seq_randomly,
        transform_image_semantic,
        transform_audio_semantic
    )
from SAD.debug import debug_var

from tqdm import tqdm
import pdb

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
        # audio.size(): size: [200, 640]
        # image.size(): 1x3x512x512

        canonical_kp = self.kpdetector_model(image)
        # tensor [canonical_kp] size: [1, 15, 3] , min: tensor(-0.8919, device='cuda:0') , max: tensor(0.9501, device='cuda:0')
        # orig -- canonical_kp['value'].size() -- [2, 15, 3]

        image_exp_pose = self.image2coffe_model(image) # image exp + pose
        # tensor [image_exp_pose] size: [1, 70] , min: tensor(-1.1567, device='cuda:0') , max: tensor(1.4598, device='cuda:0')
        # std array semantic_npy shape: (1, 70) , min: -1.1055539 , max: 1.1102859
        image_he = self.image_head_estimation(image_exp_pose)
        # image_he is dict:
        #     tensor [yaw] size: [1, 66] , min: tensor(-3.9595, device='cuda:0') , max: tensor(4.3973, device='cuda:0')
        #     tensor [pitch] size: [1, 66] , min: tensor(-4.5360, device='cuda:0') , max: tensor(5.8540, device='cuda:0')
        #     tensor [roll] size: [1, 66] , min: tensor(-3.8121, device='cuda:0') , max: tensor(6.6560, device='cuda:0')
        #     tensor [t] size: [1, 3] , min: tensor(-0.0580, device='cuda:0') , max: tensor(0.2279, device='cuda:0')
        #     tensor [exp] size: [1, 45] , min: tensor(-0.1022, device='cuda:0') , max: tensor(0.0151, device='cuda:0')

        image_kp = keypoint_transform(canonical_kp, image_he)
        # tensor [image_kp] size: [1, 15, 3] , min: tensor(-0.8479, device='cuda:0') , max: tensor(0.9343, device='cuda:0')

        num_frames = audio.shape[0]
        audio_mels = melspectrogram(audio.cpu()).to(audio.device)
        # tensor [audio_mels] size: [1, 200, 1, 80, 16] , min: tensor(-4., device='cuda:0') , max: tensor(2.5901, device='cuda:0')

        audio_ratio = get_blink_seq_randomly(num_frames)
        audio_ratio = audio_ratio.unsqueeze(0).to(audio.device) # [1, 200, 1]

        batch: Dict[str, torch.Tensor] = {}
        batch['audio_mels'] = audio_mels
        batch['image_exp_pose'] = image_exp_pose.repeat(1, num_frames, 1) # size() [1, 70] ==> [1, 200, 70]
        batch['audio_num_frames'] = num_frames
        batch['audio_ratio'] = audio_ratio
        # batch is dict:
        #     tensor [audio_mels] size: [1, 200, 1, 80, 16] , min: tensor(-4., device='cuda:0') , max: tensor(2.5901, device='cuda:0')
        #     tuple [image_exp_pose] len: 3 , torch.Size([1, 200, 70])
        #     [audio_num_frames] value: 200
        #     tensor [audio_ratio] size: [1, 200, 1] , min: tensor(0., device='cuda:0') , max: tensor(1., device='cuda:0')

        audio_exp_pose = self.audio2coffe_model(batch, pose_style=0)
        # tensor [audio_exp_pose] size: [200, 70] , min: tensor(-1.6503, device='cuda:0') , max: tensor(1.3328, device='cuda:0')


        output_list = []
        for i in tqdm(range(num_frames), 'Face Rendering'):
            audio_semantics_frame = transform_audio_semantic(audio_exp_pose[0], i) # xxxx8888
            audio_he = self.mappingnet_model(audio_semantics_frame.unsqueeze(0)) # xxxx8888
            audio_kp = keypoint_transform(canonical_kp, audio_he)

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # source_image.size() -- [2, 3, 256, 256]
            # image_kp['value'].size() -- [2, 15, 3]
            # audio_kp['value'].size() -- [2, 15, 3]

            # generator -- SADKernel(...)
            y = self.sadkernel_model(image, audio_kp=audio_kp, image_kp=image_kp)
            # y.size() -- [1, 3, 512, 512]
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            output_list.append(y.cpu())

        output = torch.cat(output_list, dim=0)
        # tensor [output] size: [200, 3, 512, 512] , min: tensor(0.1188, device='cuda:0') , max: tensor(0.9495, device='cuda:0')

        return output

    def image_head_estimation(self, image_exp_pose):
        image_semantic = transform_image_semantic(image_exp_pose).unsqueeze(0)
        # tensor [image_semantic] size: [1, 70, 27] , min: tensor(-1.1567, device='cuda:0') , max: tensor(1.4598, device='cuda:0')
        # std tensor image_semantics size: [2, 70, 27] , min: tensor(-1.0968) , max: tensor(1.1307)

        image_he = self.mappingnet_model(image_semantic)
        return image_he        