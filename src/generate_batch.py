import os

from tqdm import tqdm
import torch
import numpy as np
import random
import scipy.io as scio
import src.utils.audio as audio
import pdb

def crop_pad_audio(wav, audio_length):
    if len(wav) > audio_length:
        wav = wav[:audio_length]
    elif len(wav) < audio_length:
        wav = np.pad(wav, [0, audio_length - len(wav)], mode='constant', constant_values=0)
    return wav

def parse_audio_length(audio_length, sr, fps):
    bit_per_frames = sr / fps

    num_frames = int(audio_length / bit_per_frames)
    audio_length = int(num_frames * bit_per_frames)

    return audio_length, num_frames


def generate_blink_seq_randomly(num_frames):
    ratio = np.zeros((num_frames, 1))
    if num_frames<=20:
        return ratio
    frame_id = 0
    while frame_id in range(num_frames):
        start = random.choice(range(min(10,num_frames), min(int(num_frames/2), 70))) 
        if frame_id + start + 5 <= num_frames - 1:
            ratio[frame_id+start : frame_id+start+5, 0] = [0.5, 0.9, 1.0, 0.9, 0.5]
            frame_id = frame_id + start + 5
        else:
            break
    return ratio

def get_data(first_coeff_path, audio_path, device):
    # first_coeff_path = './results/2023_08_13_10.45.38/first_frame_dir/dell.mat'
    # audio_path = 'examples/driven_audio/chinese_news.wav'

    syncnet_mel_step_size = 16
    fps = 25

    pic_name = os.path.splitext(os.path.split(first_coeff_path)[-1])[0]
    audio_name = os.path.splitext(os.path.split(audio_path)[-1])[0]

    wav = audio.load_wav(audio_path, 16000) 
    wav_length, num_frames = parse_audio_length(len(wav), 16000, 25) # (128000, 200)
    wav = crop_pad_audio(wav, wav_length)
    orig_mel = audio.melspectrogram(wav).T # orig_mel.shape -- (641, 80)
    spec = orig_mel.copy()         # nframes 80
    indiv_mels = []

    # (Pdb) type(wav) -- <class 'numpy.ndarray'>,  (Pdb) wav.shape -- (128000,)
    # (Pdb) wav -- array([-0.00319423, -0.00569311, -0.00719347, ..., -0.14446567,
    #        -0.16598062, -0.20840327], dtype=float32)

    for i in tqdm(range(num_frames), 'mel:'):
        start_frame_num = i-2
        start_idx = int(80. * (start_frame_num / float(fps)))
        end_idx = start_idx + syncnet_mel_step_size
        seq = list(range(start_idx, end_idx))
        seq = [ min(max(item, 0), orig_mel.shape[0]-1) for item in seq ] # orig_mel.shape -- (641, 80)
        m = spec[seq, :]
        indiv_mels.append(m.T)
    indiv_mels = np.asarray(indiv_mels)         # T 80 16

    ratio = generate_blink_seq_randomly(num_frames)      # T
    source_semantics_path = first_coeff_path
    source_semantics_dict = scio.loadmat(source_semantics_path) # ./results/2023_08_13_10.45.38/first_frame_dir/dell.mat'
    ref_coeff = source_semantics_dict['coeff_3dmm'][:1,:70]   #shape -- (1, 73) ==> 1 70
    ref_coeff = np.repeat(ref_coeff, num_frames, axis=0) # ==> shape -- (200, 70)

    
    indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1).unsqueeze(0) # bs T 1 80 16

    ratio = torch.FloatTensor(ratio).unsqueeze(0)                       # bs T
    ref_coeff = torch.FloatTensor(ref_coeff).unsqueeze(0)               # bs 1 70

    indiv_mels = indiv_mels.to(device)
    ratio = ratio.to(device)
    ref_coeff = ref_coeff.to(device)

    return {'indiv_mels': indiv_mels,  # len() -- 1, indiv_mels[0].size() -- [200, 1, 80, 16]
            'ref': ref_coeff,  # size() -- [1, 200, 70]
            'num_frames': num_frames, # -- 200
            'ratio_gt': ratio, # [1, 200, 1], 
            'audio_name': audio_name, # 'chinese_news'
            'pic_name': pic_name, # 'dell'
            }

