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
from .sad import SADModel
from torchvision.transforms import Compose, ToTensor
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


def predict(audio_file, image_file, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_model()
    transform = Compose([
        lambda image: image.convert("RGB"),
        ToTensor(),
    ])

    # audio = Audio.load(audio_file)
    # audio_tensor = audio.to(device)
    audio_tensor = torch.randn(100, 27).to(device)

    image = Image.open(image_file).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # start predict
    results = []
    progress_bar = tqdm(total=100)
    # for filename in image_filenames:
    #     progress_bar.update(1)

    #     with torch.no_grad():
    #         output_tensor = model(audio_tensor, image_tensor).cpu()

    #     results.append(output_tensor)
    progress_bar.close()

    print("\n".join(results))

    todos.model.reset_device()
