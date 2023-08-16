# xxxx8888

import librosa
import librosa.filters
import numpy as np
from scipy import signal
# from src.utils.hparams import hparams as hp

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
    #  network
    # rescale=True,  # Whether to rescale audio prior to preprocessing
    # rescaling_max=0.9,  # Rescaling value
    
    # Use LWS (https://github.com/Jonathan-LeRoux/lws) for STFT and phase reconstruction
    # It"s preferred to set True to use with https://github.com/r9y9/wavenet_vocoder
    # Does not work if n_ffit is not multiple of hop_size!!
    # use_lws=False,
    
    n_fft=800,  # Extra window size is filled with 0 paddings to match this parameter
    hop_size=200,  # For 16000Hz, 200 = 12.5 ms (0.0125 * sample_rate)
    win_size=800,  # For 16000Hz, 800 = 50 ms (If None, win_size = n_fft) (0.05 * sample_rate)
    sample_rate=16000,  # 16000Hz (corresponding to librispeech) (sox --i <filename>)
    
    # frame_shift_ms=None,  # Can replace hop_size parameter. (Recommended: 12.5)
    
    # # Mel and Linear spectrograms normalization/scaling and clipping
    # signal_normalization=True,
    # # Whether to normalize mel spectrograms to some predefined range (following below parameters)
    # allow_clipping_in_normalization=True,  # Only relevant if mel_normalization = True
    # symmetric_mels=True,
    # Whether to scale the data to be symmetric around 0. (Also multiplies the output range by 2, 
    # faster and cleaner convergence)
    max_abs_value=4.,
    # max absolute value of data. If symmetric, data will be [-max, max] else [0, max] (Must not 
    # be too big to avoid gradient explosion, 
    # not too small for fast convergence)
    # Contribution by @begeekmyfriend
    # Spectrogram Pre-Emphasis (Lfilter: Reduce spectrogram noise and helps model certitude 
    # levels. Also allows for better G&L phase reconstruction)
    # preemphasize=True,  # whether to apply filter
    preemphasis=0.97,  # filter coefficient.
    
    # Limits
    min_level_db=-100,
    ref_level_db=20,
    fmin=55,
    # Set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To 
    # test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
    fmax=7600,  # To be increased/reduced depending on data.

    # ###################### Our training parameters #################################
    # img_size=96,
    fps=25,
    
    # batch_size=16,
    # initial_learning_rate=1e-4,
    # nepochs=300000,  ### ctrl + c, stop whenever eval loss is consistently greater than train loss for ~10 epochs
    # num_workers=20,
    # checkpoint_interval=3000,
    # eval_interval=3000,
    # writer_interval=300,
    # save_optimizer_state=True,

    # syncnet_wt=0.0, # is initially zero, will be set automatically to 0.03 later. Leads to faster convergence. 
    # syncnet_batch_size=64,
    # syncnet_lr=1e-4,
    # syncnet_eval_interval=1000,
    # syncnet_checkpoint_interval=10000,

    # disc_wt=0.07,
    # disc_initial_learning_rate=1e-4,
)
