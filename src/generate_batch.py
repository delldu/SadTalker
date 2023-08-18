import os

from tqdm import tqdm
import torch
import numpy as np
import random
import scipy.io as scio
import src.utils.audio as audio
from src.utils.debug import debug_var
import pdb


def crop_pad_audio(wav, audio_length):
    if len(wav) > audio_length:
        wav = wav[:audio_length]
    elif len(wav) < audio_length:
        wav = np.pad(wav, [0, audio_length - len(wav)], mode='constant', constant_values=0)
    return wav

def parse_audio_length(audio_length, sr, fps):
    # sr = 16000, fps=25
    bit_per_frames = sr / fps # ==> 640

    audio_num_frames = int(audio_length / bit_per_frames)
    audio_length = int(audio_num_frames * bit_per_frames)

    return audio_length, audio_num_frames

def generate_blink_seq_randomly(audio_num_frames):
    audio_ratio = np.zeros((audio_num_frames, 1))
    if audio_num_frames <= 20:
        return audio_ratio
    frame_id = 0
    while frame_id in range(audio_num_frames):
        start = random.choice(range(min(10,audio_num_frames), min(int(audio_num_frames/2), 70))) 
        if frame_id + start + 5 <= audio_num_frames - 1:
            audio_ratio[frame_id+start : frame_id+start+5, 0] = [0.5, 0.9, 1.0, 0.9, 0.5]
            frame_id = frame_id + start + 5
        else:
            break
    return audio_ratio

def get_data(image_coeff_path, audio_path, device):
    # image_coeff_path = './results/2023_08_13_10.45.38/first_frame_dir/dell.mat'
    # audio_path = 'examples/driven_audio/chinese_news.wav'

    mel_step_size = 16
    fps = 25

    image_name = os.path.splitext(os.path.split(image_coeff_path)[-1])[0]
    audio_name = os.path.splitext(os.path.split(audio_path)[-1])[0]

    # xxxx9999 Step 1
    wav = audio.load_wav(audio_path, 16000) # wav.dtype -- dtype=float32
    # wav.reshape(200, -1).shape -- (200, 640)

    wav_length, audio_num_frames = parse_audio_length(len(wav), 16000, 25) # (128000, 200)
    wav = crop_pad_audio(wav, wav_length)
    # array wav shape: (128000,) , min: -1.0112159 , max: 1.0876185

    orig_mel = audio.melspectrogram(wav).T # orig_mel.shape -- (641, 80), xxxx8888 !!!
    # array orig_mel shape: (641, 80) , min: -4.0 , max: 2.5998246882359766

    torch_mel = audio.torch_melspectrogram(torch.from_numpy(wav)).T
    debug_var("torch_mel", torch_mel)

    audio_mels = []
    for i in tqdm(range(audio_num_frames), 'mel:'):
        start_frame_num = i-2
        start_idx = int(80. * (start_frame_num / float(fps)))
        end_idx = start_idx + mel_step_size
        seq = list(range(start_idx, end_idx))
        seq = [ min(max(item, 0), orig_mel.shape[0]-1) for item in seq ] # orig_mel.shape -- (641, 80)
        m = orig_mel[seq, :]
        audio_mels.append(m.T)
    audio_mels = np.asarray(audio_mels) # ==> (200, 80, 16)
    audio_mels = torch.FloatTensor(audio_mels).unsqueeze(1).unsqueeze(0) # bs T 1 80 16
    audio_mels = audio_mels.to(device) # size() -- [1, 200, 1, 80, 16]

    audio_ratio = generate_blink_seq_randomly(audio_num_frames)
    audio_ratio = torch.FloatTensor(audio_ratio).unsqueeze(0).to(device) # [1, 200, 1]

    source_semantics_path = image_coeff_path
    image_coeff_dict = scio.loadmat(source_semantics_path) # ./results/2023_08_13_10.45.38/first_frame_dir/dell.mat'

    # 'coeff_3dmm' -- exp -- 64, angle -- 3, face trans -- 3, whole image trans -- 3 ==> total dim 73
    image_exp_pose = image_coeff_dict['coeff_3dmm'][:1,:70]   #shape -- (1, 73) ==> 1 70
    image_exp_pose = np.repeat(image_exp_pose, audio_num_frames, axis=0) # ==> shape -- (200, 70)
    image_exp_pose = torch.FloatTensor(image_exp_pose).unsqueeze(0).to(device) # [1, 200, 70]

    output = {'audio_mels': audio_mels,  # size() -- [1, 200, 1, 80, 16]
            'image_exp_pose': image_exp_pose,  # size() -- [1, 200, 70]
            'audio_num_frames': audio_num_frames, # -- 200
            'audio_ratio': audio_ratio, # [1, 200, 1], 
            'audio_name': audio_name, # 'chinese_news'
            'image_name': image_name, # 'dell'
            }

    # debug_var("get_data.output", output)
    # get_data.output is dict:
    #     tensor audio_mels size: [1, 200, 1, 80, 16] , min: tensor(-4., device='cuda:0') , max: tensor(2.5998, device='cuda:0')
    #     tensor image_exp_pose size: [1, 200, 70] , min: tensor(-1.0968, device='cuda:0') , max: tensor(1.1307, device='cuda:0')
    #     audio_num_frames value: 200
    #     tensor audio_ratio size: [1, 200, 1] , min: tensor(0., device='cuda:0') , max: tensor(1., device='cuda:0')
    #     audio_name value: 'chinese_news'
    #     image_name value: 'dell'

    return output
