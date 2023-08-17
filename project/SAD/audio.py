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
import math
import random
import torch
import torchaudio
from SAD.debug import debug_var
import pdb

def melspectrogram(wav):
    B = wav.shape[0]
    wav = torch_preemphasis(wav.reshape(-1), hp.preemphasis) # preemphasis -- 0.97
    T = torchaudio.transforms.Spectrogram(
            n_fft=hp.n_fft, win_length=hp.win_size, hop_length=hp.hop_size,
            power=1, normalized=False, # !!! here two items configuration is very import !!!
        )
    D = T(wav)
    M = torch_linear_to_mel(D)

    S = torch_amp_to_db(M) - hp.ref_level_db # hp.ref_level_db -- 20
    orig_mel = torch_normalize(S).transpose(1, 0) # size() -- [641, 80]

    mel_step_size = 16

    mels_list = []
    for i in range(B):
        start_frame_num = i-2
        start_idx = int(hp.num_mels * (start_frame_num / float(hp.fps))) # hp.fps = 25
        end_idx = start_idx + mel_step_size
        seq = list(range(start_idx, end_idx))
        seq = [ min(max(item, 0), orig_mel.shape[0] - 1) for item in seq ] # orig_mel.shape -- (641, 80)
        m = orig_mel[seq, :]
        mels_list.append(m.transpose(1, 0))
    # mels[0] size() -- [80, 16]
    mels = torch.stack(mels_list, dim=0)
    mels = mels.unsqueeze(1).unsqueeze(0) # size() -- [1, 200, 1, 80, 16]
    return mels


def get_blink_seq_randomly(num_frames):
    ratio = torch.zeros((num_frames, 1))
    if num_frames <= 20:
        return ratio
    frame_id = 0
    while frame_id in range(num_frames):
        # random ?????? torch.jit.script ???
        start = random.choice(range(min(10, num_frames), min(int(num_frames/2), 70))) 
        if frame_id + start + 5 <= num_frames - 1:
            ratio[frame_id+start : frame_id+start+5, 0] = torch.Tensor([0.5, 0.9, 1.0, 0.9, 0.5])
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
    audio_num_frames = coeff_3dmm.shape[0]
    seq = list(range(frame_index- semantic_radius, frame_index + semantic_radius+1))
    # seq -- [-13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 
    #     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    index = [ min(max(item, 0), audio_num_frames-1) for item in seq ] 
    # index -- [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    coeff_3dmm_g = coeff_3dmm[index, :] # shape -- (27, 70)
    return coeff_3dmm_g.transpose(1,0) # ==> shape -- (70, 27)


def torch_preemphasis(wav, c):
    return torchaudio.functional.preemphasis(wav, coeff = c)

def torch_linear_to_mel(spectogram):
    mel_filters = torchaudio.functional.melscale_fbanks(
        int(hp.n_fft // 2 + 1),
        n_mels=hp.num_mels,
        f_min=hp.fmin,
        f_max=hp.fmax,
        sample_rate=hp.sample_rate,
        norm="slaney",
    )

    return torch.mm(mel_filters.T, spectogram)


def torch_amp_to_db(x):
    min_level = math.exp(hp.min_level_db / 20.0 * math.log(10.0))
    return 20.0 * torch.log10(torch.maximum(torch.Tensor([min_level]), x))

def torch_normalize(S):
    x = (2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value    
    return torch.clip(x, -hp.max_abs_value, hp.max_abs_value)


class HParams:
    def __init__(self, **kwargs):
        self.data = {}

        for key, value in kwargs.items():
            self.data[key] = value

    def __getattr__(self, key):
        if key not in self.data:
            raise AttributeError("'HParams' object has no attribute %s" % key)
        return self.data[key]

    def set_hparam(self, key, value):
        self.data[key] = value


# Default hyperparameters
hp = HParams(
    fps=25,
    num_mels=80,  # Number of mel-spectrogram channels and local conditioning dimensionality
    n_fft=800,  # Extra window size is filled with 0 paddings to match this parameter
    hop_size=200,  # For 16000Hz, 200 = 12.5 ms (0.0125 * sample_rate)
    win_size=800,  # For 16000Hz, 800 = 50 ms (If None, win_size = n_fft) (0.05 * sample_rate)
    sample_rate=16000,  # 16000Hz (corresponding to librispeech) (sox --i <filename>)
    max_abs_value=4.,
    preemphasis=0.97,  # filter coefficient.
    
    # Limits
    min_level_db=-100,
    ref_level_db=20,
    fmin=55,
    fmax=7600,  # To be increased/reduced depending on data.
)
