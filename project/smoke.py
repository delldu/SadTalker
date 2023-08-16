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

import pdb
import os
import time
import torch
import SAD

from tqdm import tqdm

if __name__ == "__main__":
    model, device = SAD.get_model()

    N = 100
    B, C, H, W = 1, 3, 512, 512

    mean_time = 0
    progress_bar = tqdm(total=N)
    for count in range(N):
        progress_bar.update(1)

        audio_tensor = torch.randn(B, 70, 27)
        image_tensor = torch.randn(B, C, H, W)
        # print("image_tensor: ", image_tensor.size())

        start_time = time.time()
        with torch.no_grad():
            y = model(audio_tensor.to(device), image_tensor.to(device))
        torch.cuda.synchronize()
        mean_time += time.time() - start_time

    mean_time /= N
    print(f"Mean spend {mean_time:0.4f} seconds")
    os.system("nvidia-smi | grep python")
