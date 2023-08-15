# xxxx8888

import librosa
import librosa.filters
import numpy as np
from scipy import signal
from src.utils.hparams import hparams as hp

import math
import torch
import torchaudio

from src.utils.debug import debug_var
import pdb

def load_wav(path, sr):
    return librosa.core.load(path, sr=sr)[0]

def preemphasis(wav, k):
    return signal.lfilter([1, -k], [1], wav)

def melspectrogram(wav):
    D = _stft(preemphasis(wav, hp.preemphasis)) # preemphasis -- 0.97
    # array D shape: (401, 641) , min: (-25.584568141758577-9.771092532498667j) , max: (33.762999171308316+19.310081052981108j)
    # array D.abs shape: (401, 641) , min: 7.868068141458906e-09 , max: 38.89497838172

    M = _linear_to_mel(np.abs(D))
    # array M shape: (80, 641) , min: 2.181826502756544e-05 , max: 1.3331850352656407

    S = _amp_to_db(M) - hp.ref_level_db # hp.ref_level_db -- 20
    N = _normalize(S)
    # array N shape: (80, 641) , min: -4.0 , max: 2.5998246882359766

    return N

def _stft(y):
    return librosa.stft(y=y, n_fft=hp.n_fft, hop_length=hp.hop_size, win_length=hp.win_size)

# Conversions
_mel_basis = None

def _linear_to_mel(spectogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()

    return np.dot(_mel_basis, spectogram)

def _build_mel_basis():
    assert hp.fmax <= hp.sample_rate // 2
    return librosa.filters.mel(sr=hp.sample_rate, # 16000
                               n_fft=hp.n_fft, # 800
                               n_mels=hp.num_mels, # 80
                               fmin=hp.fmin, # 55
                               fmax=hp.fmax) # 7600

def _amp_to_db(x):
    min_level = np.exp(hp.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))

def _normalize(S):
    # hp.max_abs_value -- 4

    x = (2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value    
    return np.clip(x, -hp.max_abs_value, hp.max_abs_value)



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
