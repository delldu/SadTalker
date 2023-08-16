"""SadTalking Model Package."""
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

__version__ = "1.0.0"

import os
from tqdm import tqdm
from PIL import Image
import numpy as np

import torch
import todos
import torchaudio
from torchvision.transforms import Compose, ToTensor
from SAD.sad import SADModel
from SAD.debug import debug_var

import pdb


def create_model():
    """
    Create model
    """

    model = SADModel()
    device = todos.model.get_device()
    model = model.to(device)
    model.eval()
    print(f"Running model on {device} ...")

    return model, device


def get_model():
    """Load jit script model."""

    model, device = create_model()
    # print(model)

    # model = torch.jit.script(model)
    # todos.data.mkdir("output")
    # if not os.path.exists("output/SAD.torch"):
    #     model.save("output/SAD.torch")

    return model, device

def load_wav(audio_path, new_sample_rate=16000):
    wav, sample_rate = torchaudio.load(audio_path)
    wav = torchaudio.functional.resample(wav, orig_freq=sample_rate, new_freq=new_sample_rate)[0]

    # Parse audio length
    fps = 25
    bit_per_frames = new_sample_rate / fps
    num_frames = int(len(wav) / bit_per_frames)
    wav_length = int(num_frames * bit_per_frames)

    # Crop pad
    if len(wav) > wav_length:
        wav = wav[:wav_length]
    elif len(wav) < wav_length:
        wav = torch.pad(wav, [0, wav_length - len(wav)], mode='constant', constant_values=0)

    return wav.reshape(num_frames, -1)


def predict(audio_file, image_file, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_model()

    # load audio
    audio = load_wav(audio_file)
    audio_tensor = audio.to(device)
    # tensor [audio_tensor] size: [200, 640] , min: tensor(-1.0130, device='cuda:0') , max: tensor(1.0737, device='cuda:0')

    image = Image.open(image_file).convert("RGB")
    image_tensor = ToTensor()(image).unsqueeze(0).to(device)
    # tensor [image_tensor] size: [1, 3, 512, 512] , min: tensor(0.1176, device='cuda:0') , max: tensor(1., device='cuda:0')

    with torch.no_grad():
        output_tensor = model(audio_tensor, image_tensor)

    todos.data.mkdir(output_dir)
    B, C, H, W = output_tensor.size()
    for i in range(B):
        output_file = f"{output_dir}/{i:06d}.png"
        todos.data.save_tensor([output_tensor[i].unsqueeze(0)], output_file)

    todos.model.reset_device()


