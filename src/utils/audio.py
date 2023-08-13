# xxxx8888

import librosa
import librosa.filters
import numpy as np
from scipy import signal # xxxx8888
from src.utils.hparams import hparams as hp

def load_wav(path, sr):
    return librosa.core.load(path, sr=sr)[0]

def preemphasis(wav, k):
    return signal.lfilter([1, -k], [1], wav)

def melspectrogram(wav):
    D = _stft(preemphasis(wav, hp.preemphasis))
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - hp.ref_level_db
    return _normalize(S)

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
    return librosa.filters.mel(sr=hp.sample_rate, 
                               n_fft=hp.n_fft, 
                               n_mels=hp.num_mels,
                               fmin=hp.fmin, 
                               fmax=hp.fmax)

def _amp_to_db(x):
    min_level = np.exp(hp.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))

def _normalize(S):
    x = (2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value    
    return np.clip(x, -hp.max_abs_value, hp.max_abs_value)
