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
from SAD.keypoint_detector import KPDetector
from SAD.mapping import MappingNet
from SAD.util import load_weights, keypoint_transform

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

def transform_image_semantic(coeff_3dmm, semantic_radius: int=13):
    # semantic_radius note: 2-13 is lower power and important, others is almost noise !!!
    # coeff_3dmm shape: (1, 70) , min: -1.0967898 , max: 1.13074
    coeff_3dmm_list =  [coeff_3dmm for i in range(0, semantic_radius*2+1)]
    coeff_3dmm_g = torch.cat(coeff_3dmm_list, 0) # shape: (27, 70)
    return coeff_3dmm_g.transpose(1, 0) # ==> shape: (70, 27)

def transform_audio_semantic(coeff_3dmm, frame_index: int, semantic_radius: int=13):
    # coeff_3dmm shape: (200, 70) , min: -1.6095467 , max: 1.0893884
    num_frames = coeff_3dmm.shape[0]
    seq = list(range(frame_index- semantic_radius, frame_index + semantic_radius+1))
    # seq -- [-13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 
    #     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    index = [ min(max(item, 0), num_frames-1) for item in seq ] 
    # index -- [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    coeff_3dmm_g = coeff_3dmm[index, :] # shape -- (27, 70)
    return coeff_3dmm_g.transpose(1,0) # ==> shape -- (70, 27)


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

        load_weights(self, model_path="models/SAD.pth") # xxxx8888

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
        # audio.size(): size: [200, 640]
        # image.size(): 1x3x512x512

        canonical_kp = self.kpdetector_model(image)
        # tensor [canonical_kp] size: [1, 15, 3] , min: tensor(-0.8919, device='cuda:0') , max: tensor(0.9501, device='cuda:0')

        image_exp_pose = self.image2coffe_model(image) # image exp + pose
        # tensor [image_exp_pose] size: [1, 70] , min: tensor(-1.1567, device='cuda:0') , max: tensor(1.4598, device='cuda:0')
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
        audio_mels = self.get_mel_spectrogram(audio.cpu()).to(audio.device)
        audio_ratio = get_blink_seq_randomly(num_frames)
        audio_ratio = audio_ratio.unsqueeze(0).to(audio.device) # [1, 200, 1]

        batch: Dict[str, torch.Tensor] = {}
        batch['audio_mels'] = audio_mels
        batch['image_exp_pose'] = image_exp_pose.repeat(1, num_frames, 1) # size() [1, 70] ==> [1, 200, 70]
        # batch['num_frames'] = num_frames
        batch['audio_ratio'] = audio_ratio
        # batch is dict:
        #     tensor [audio_mels] size: [1, 200, 1, 80, 16] , min: tensor(-4., device='cuda:0') , max: tensor(2.5901, device='cuda:0')
        #     tuple [image_exp_pose] len: 3 , torch.Size([1, 200, 70])
        ##    [num_frames] value: 200
        #     tensor [audio_ratio] size: [1, 200, 1] , min: tensor(0., device='cuda:0') , max: tensor(1., device='cuda:0')

        audio_exp_pose = self.audio2coffe_model(batch, pose_style=0).squeeze(0)
        # tensor [audio_exp_pose] size: [200, 70] , min: tensor(-1.6503, device='cuda:0') , max: tensor(1.3328, device='cuda:0')

        output_list = []
        # for i in tqdm(range(num_frames), 'Face Rendering'):
        for i in range(num_frames):
            frame_semantics = transform_audio_semantic(audio_exp_pose, i).unsqueeze(0) # size() -- [1, 70, 27]
            audio_he = self.mappingnet_model(frame_semantics)
            audio_kp = keypoint_transform(canonical_kp, audio_he)

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
        # tensor [output] size: [200, 3, 512, 512] , min: tensor(0.1188, device='cuda:0') , max: tensor(0.9495, device='cuda:0')

        return output

    def image_head_estimation(self, image_exp_pose):
        image_semantics = transform_image_semantic(image_exp_pose).unsqueeze(0)
        # tensor [image_semantics] size: [1, 70, 27] , min: tensor(-1.1567, device='cuda:0') , max: tensor(1.4598, device='cuda:0')

        image_he = self.mappingnet_model(image_semantics)
        return image_he
