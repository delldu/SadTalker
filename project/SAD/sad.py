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


import random
import torch
from torch import nn
import torchaudio

from SAD.image2coeff import Image2Coeff
from SAD.audio2coeff import Audio2Coeff
from SAD.sadkernel import SADKernel
from SAD.keypoint_detector import KeypointDetector
from SAD.mapping import MappingNet
from SAD.util import load_weights
#, keypoint_transform
from typing import Tuple
from tqdm import tqdm
import todos
import pdb


def get_blink_seq_randomly(num_frames: int):
    ratio = torch.zeros((num_frames, 1))
    if num_frames <= 20:
        return ratio

    frame_id = 0
    for frame_id in range(num_frames):
        # random ?????? torch.jit.script ???
        # start = random.choice(range(min(10, num_frames), min(int(num_frames/2), 70))) 
        start = torch.randint(min(10, num_frames), min(int(num_frames/2), 70), (1,)).item()
        if frame_id + start + 5 <= num_frames - 1:
            ratio[frame_id+start : frame_id+start+5, 0] = torch.tensor([0.5, 0.9, 1.0, 0.9, 0.5])
            frame_id = frame_id + start + 5
        else:
            break
    return ratio

def transform_image_semantic(image_exp_pose, semantic_radius: int=13):
    # tensor [image_exp_pose] size: [1, 70], min: -2.225995, max: 0.35268, mean: -0.052696
    # semantic_radius note: 2-13 is lower power and important, others is almost noise !!!
    coeff_3dmm_list =  [image_exp_pose for i in range(0, semantic_radius*2+1)]
    coeff_3dmm_g = torch.cat(coeff_3dmm_list, dim=0) # shape: (27, 70)
    return coeff_3dmm_g.transpose(1, 0) # ==> shape: (70, 27)

def transform_audio_semantic(audio_exp_pose, frame_index: int, semantic_radius: int=13):
    # tensor [audio_exp_pose] size: [200, 70], min: -2.489626, max: 0.524471, mean: -0.075972
    num_frames = audio_exp_pose.shape[0]
    seq = list(range(frame_index- semantic_radius, frame_index + semantic_radius+1))
    # seq -- [-13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 
    #     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    index = [ min(max(item, 0), num_frames-1) for item in seq ] 
    # index -- [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    coeff_3dmm_g = audio_exp_pose[index, :] # shape -- (27, 70)
    return coeff_3dmm_g.transpose(1, 0) # ==> shape -- (70, 27)


class SADModel(nn.Module):
    def __init__(self):
        r""" 
        SadTalker: Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven 
            Single Image Talking Face Animation
        """
        super().__init__()
        self.image2coffe_model = Image2Coeff() # image_pose_exp.onnx
        self.audio2coffe_model = Audio2Coeff() # audio_pose_exp.onnx
        self.sadkernel_model = SADKernel() # audio_face_render.onnx
        self.kpdetector_model = KeypointDetector()
        self.mappingnet_model = MappingNet()

        load_weights(self, model_path="models/SAD.pth")



    def get_mel_spectrogram(self, wav):
        '''
        wav must be BxS format !!!
        Configuration:
            fps=25,
            num_mels=80,  # Number of mel-spectrogram channels and local conditioning dimensionality
            n_fft=800,  # Extra window size is filled with 0 paddings to match this parameter
            hop_length=200,  # For 16000Hz, 200 = 12.5 ms (0.0125 * sample_rate)
            win_length=800,  # For 16000Hz, 800 = 50 ms (If None, win_length = n_fft) (0.05 * sample_rate)
            sample_rate=16000,  # 16000Hz (corresponding to librispeech) (sox --i <filename>)
            max_abs_value=4.,
            preemphasis=0.97,  # filter coefficient.
            
            # Limits
            min_level_db=-100,
            ref_level_db=20,
            fmin=55.0,
            fmax=7600.0,  # To be increased/reduced depending on data.

        '''
        B = wav.shape[0]
        wav = torchaudio.functional.preemphasis(wav, coeff = 0.97) # 0.97 -- hp.preemphasis

        # Mel Filter Bank, generates the filter bank for converting frequency bins to mel-scale bins
        mel_filters = torchaudio.functional.melscale_fbanks(
            401, # int(hp.n_fft // 2 + 1),
            n_mels=80, # hp.num_mels,
            f_min=55.0, #hp.fmin,
            f_max=7600.0, # hp.fmax,
            sample_rate=16000, # hp.sample_rate,
            norm="slaney",
        )
        D = torchaudio.functional.spectrogram(
            wav.reshape(-1), # should 1-D data
            pad=0, window=torch.hann_window(800),
            n_fft=800, win_length=800, hop_length=200,
            power=1.0, normalized=False, # !!! here two items configuration is very import !!!
        ) # ==> [401, 641]
        S = torch.mm(mel_filters.T, D) # mel_filters.T.size() -- [80, 401]

        # Amp to DB
        # min_level = math.exp(hp.min_level_db / 20.0 * math.log(10.0))
        # return 20.0 * torch.log10(torch.maximum(torch.Tensor([min_level]), x)) - hp.min_level_db
        S = torch.clamp(S, 9.9999e-06)
        S = 20.0 * torch.log10(S) - 20.0

        # normalize
        # S = (2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value    
        # S =torch.clip(S, -hp.max_abs_value, hp.max_abs_value)
        S = 8.0 * ((S + 100.0) / 100.0) - 4.0
        orig_mel = torch.clip(S, -4.0, 4.0).transpose(1, 0) # size() -- [641, 80]

        mels_list = []
        mel_step_size = 16
        for i in range(B):
            start_frame_num = i - 2
            start_idx = int(80 * (start_frame_num/25.0)) #hp.num_mels -- 80, hp.fps = 25.0
            end_idx = start_idx + mel_step_size
            seq = list(range(start_idx, end_idx))
            seq = [ min(max(item, 0), orig_mel.shape[0] - 1) for item in seq ] # orig_mel.shape -- (641, 80)
            m = orig_mel[seq, :]
            mels_list.append(m.transpose(1, 0))
        # mels[0] size() -- [80, 16]
        mels = torch.stack(mels_list, dim=0)
        mels = mels.unsqueeze(1).unsqueeze(0) # size() -- [1, 200, 1, 80, 16]
        return mels

    def forward(self, audio, image):
        # tensor [audio] size: [200, 640], min: -1.013043, max: 1.073747, mean: -8.6e-05
        # tensor [image] size: [1, 3, 512, 512], min: 0.117647, max: 1.0, mean: 0.644081

        canonical_kp = self.kpdetector_model(image)
        # tensor [canonical_kp] size: [1, 15, 3], min: -0.891859, max: 0.950069, mean: 0.015366

        image_exp_pose = self.image2coffe_model(image) # image exp + pose
        # tensor [image_exp_pose] size: [1, 70], min: -1.156697, max: 1.459776, mean: 0.023419

        # image_he
        # yaw, pitch, roll, trans, exp = self.image_head_estimation(image_exp_pose)
        image_semantics = transform_image_semantic(image_exp_pose)
        # tensor [image_semantics] size: [70, 27] , min: tensor(-1.1567, device='cuda:0') , max: tensor(1.4598, device='cuda:0')

        # (yaw, pitch, roll, trans, exp) = self.mappingnet_model(image_semantics)


        # tensor [yaw] size: [1, 66], min: -3.959461, max: 4.397331, mean: 0.079592
        # tensor [pitch] size: [1, 66], min: -4.535993, max: 5.854048, mean: -0.505516
        # tensor [roll] size: [1, 66], min: -3.812059, max: 6.655958, mean: -0.263339
        # tensor [trans] size: [1, 3], min: -0.058004, max: 0.227935, mean: 0.068895
        # tensor [exp] size: [1, 45], min: -0.102243, max: 0.015078, mean: -0.002095
        image_kp = self.mappingnet_model(canonical_kp, image_semantics) # fine tunning for canonical_kp
        # tensor [image_kp] size: [1, 15, 3], min: -0.847928, max: 0.93429, mean: 0.040402

        num_frames = audio.shape[0]
        audio_mels = self.get_mel_spectrogram(audio.cpu()).to(audio.device)
        audio_ratio = get_blink_seq_randomly(num_frames)
        audio_ratio = audio_ratio.unsqueeze(0).to(audio.device) # [1, 200, 1]

        # tensor [audio_mels] size: [1, 200, 1, 80, 16], min: -4.0, max: 2.590095, mean: -1.017794
        # tensor [image_exp_pose] size: [1, 200, 70], min: -1.156697, max: 1.459776, mean: 0.023419
        # tensor [audio_ratio] size: [1, 200, 1], min: 0.0, max: 1.0, mean: 0.6575
        image_exp_pose = image_exp_pose.repeat(1, num_frames, 1)
        audio_exp_pose = self.audio2coffe_model(audio_mels, image_exp_pose, audio_ratio, 
            pose_style=10).squeeze(0) # xxxx_8888
        # tensor [audio_exp_pose] size: [200, 70], min: -1.703708, max: 1.255959, mean: -0.02074

        output_list = []
        for i in tqdm(range(num_frames), 'Rendering'):
            audio_semantics = transform_audio_semantic(audio_exp_pose, i)
            # size() -- [70, 27]

            # audio_he
            # yaw, pitch, roll, trans, exp = self.mappingnet_model(audio_semantics)
            audio_kp = self.mappingnet_model(canonical_kp, audio_semantics)

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # size() -- [1, 3, 512, 512]
            # audio_kp.size() -- [1, 15, 3]
            # image_kp.size() -- [1, 15, 3]

            # generator -- SADKernel(...)
            y = self.sadkernel_model(image, audio_kp=audio_kp, image_kp=image_kp)
            # y.size() -- [1, 3, 512, 512]
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            output_list.append(y.cpu())

        output = torch.cat(output_list, dim=0)

        # tensor [output] size: [200, 3, 512, 512], min: 0.121079, max: 0.954297, mean: 0.62631
        return output

    # def image_head_estimation(self, image_exp_pose)->Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    #     image_semantics = transform_image_semantic(image_exp_pose).unsqueeze(0)
    #     # tensor [image_semantics] size: [1, 70, 27] , min: tensor(-1.1567, device='cuda:0') , max: tensor(1.4598, device='cuda:0')

    #     # (yaw, pitch, roll, trans, exp) = self.mappingnet_model(image_semantics)
    #     # return (yaw, pitch, roll, trans, exp)
    #     return self.mappingnet_model(image_semantics)