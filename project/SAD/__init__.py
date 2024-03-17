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
import subprocess
from tqdm import tqdm
from PIL import Image

import torch
import todos
import torchaudio
from torchvision.transforms import Compose, ToTensor
from SAD.sad import SADModel
import todos
import pdb


def runcmd(cmd):
    try:
        if 'DEBUG' in os.environ:
            print("Run command:", " ".join(cmd))
        with subprocess.Popen(cmd, stdout=subprocess.PIPE) as proc:
            print(proc.stdout.read().decode(encoding="utf-8"))
    except (OSError, ValueError):
        print(f"video encode '{input_dir}'' error.")
        return False
    return True


def video_encode(input_dir, output_file):
    if not os.path.exists(input_dir):
        print(f"Dir {input_dir} not exists.")
        return False

    # ffmpeg -i output/%6d.png -vcodec png -pix_fmt rgba -y blackswan.mp4
    cmd = [
        "ffmpeg", "-y", "-i", f"{input_dir}/%6d.png", "-vcodec", "png", "-pix_fmt", "rgba", output_file,
    ]
    return runcmd(cmd)

def video_merge(input_video_file, input_audio_file, output_file):
    if not os.path.exists(input_video_file):
        print(f"File {input_video_file} not exists.")
        return False

    if not os.path.exists(input_audio_file):
        print(f"File {input_audio_file} not exists.")
        return False

    # cmd = r'ffmpeg -y -i "%s" -i "%s" -vcodec copy "%s"' % (video, audio, temp_file)
    cmd = [
        "ffmpeg", "-y", "-i", input_video_file, "-i", input_audio_file, "-vcodec", "copy", output_file,
    ]
    return runcmd(cmd)


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

    # https://github.com/pytorch/pytorch/issues/52286
    torch._C._jit_set_profiling_executor(False)
    # C++ Reference
    # torch::jit::getProfilingMode() = false;                                                                                                             
    # torch::jit::setTensorExprFuserEnabled(false);

    model = torch.jit.script(model)
    # model = torch.jit.freeze(model)
    # model = torch.jit.optimize_for_inference(model)
    
    todos.data.mkdir("output")
    if not os.path.exists("output/SAD.torch"):
        model.save("output/SAD.torch")

    return model, device

def load_wav(audio_path, new_sample_rate=16000):
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=new_sample_rate)
    waveform = waveform[0] # half enough for application

    # Parse audio length
    fps = 25
    bit_per_frames = new_sample_rate / fps
    num_frames = int(len(waveform) / bit_per_frames)
    wav_length = int(num_frames * bit_per_frames)

    # Crop pad
    if len(waveform) > wav_length:
        waveform = waveform[:wav_length]
    elif len(waveform) < wav_length:
        waveform = torch.pad(waveform, [0, wav_length - len(waveform)], mode='constant', constant_values=0)

    return waveform.reshape(num_frames, -1)


def predict(audio_file, image_file, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_model()

    # load audio
    audio = load_wav(audio_file)
    audio_tensor = audio.to(device)
    # tensor [audio_tensor] size: [200, 640] , min: -1.0130 , max: 1.0737

    image = Image.open(image_file).convert("RGB").resize((512, 512))
    image_tensor = ToTensor()(image).unsqueeze(0).to(device)
    # tensor [image_tensor] size: [1, 3, 512, 512] , min: 0.1176, max: 1.

    with torch.no_grad():
        output_tensor = model(audio_tensor, image_tensor)

    todos.data.mkdir(output_dir)
    B, C, H, W = output_tensor.size()
    for i in range(B):
        output_file = f"{output_dir}/{i:06d}.png"
        todos.data.save_tensor([output_tensor[i].unsqueeze(0)], output_file)

    video_encode(f"{output_dir}", f"{output_dir}/sad_video.mp4")
    video_merge(f"{output_dir}/sad_video.mp4", audio_file, f"{output_dir}/sad.mp4")

    todos.model.reset_device()


