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
import torch
import torchaudio

from .debug import debug_var
import pdb

def torch_load_wav(audio_path, new_sample_rate=16000):
    waveform, sample_rate = torchaudio.load(audio_path)
    return torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=new_sample_rate)[0]

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

def torch_melspectrogram(wav):
    wav = torch_preemphasis(wav, hp.preemphasis) # preemphasis -- 0.97
    T = torchaudio.transforms.Spectrogram(
            n_fft=hp.n_fft, win_length=hp.win_size, hop_length=hp.hop_size,
            power=1, normalized=False, # !!! here two items configuration is very import !!!
        )
    D = T(wav)
    M = torch_linear_to_mel(D)

    S = torch_amp_to_db(M) - hp.ref_level_db # hp.ref_level_db -- 20
    N = torch_normalize(S)

    return N


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
