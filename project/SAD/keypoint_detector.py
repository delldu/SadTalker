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
import torch
from torch import nn
import torch.nn.functional as F
from SAD.util import (
    DownBlock2d,
    UpBlock3d, 
    make_coordinate_grid, 
    load_weights,
)


from typing import Dict

import pdb

class KPDetector(nn.Module):
    """
    src/config/facerender.yml
      common_params:
        num_kp: 15 
        image_channel: 3                    
        feature_channel: 32
        estimate_jacobian: False   # True

      kp_detector_params:
         temperature: 0.1
         block_expansion: 32            
         max_features: 1024
         scale_factor: 0.25         # 0.25
         num_blocks: 5
         reshape_channel: 16384  # 16384 = 1024 * 16
         reshape_depth: 16
    """
    def __init__(self, 
            block_expansion=32, 
            feature_channel=32, 
            num_kp=15, 
            image_channel=3, 
            max_features=1024, 
            reshape_channel=16384, 
            reshape_depth=16, 
            num_blocks=5, 
            temperature=0.1, 
            scale_factor=0.25,
            # estimate_jacobian=False, 
        ):
        super(KPDetector, self).__init__()
        self.predictor = KPHourglass(block_expansion, in_features=image_channel,
                                     max_features=max_features, reshape_features=reshape_channel,
                                     reshape_depth=reshape_depth, num_blocks=num_blocks)

        self.kp = nn.Conv3d(in_channels=self.predictor.out_filters, out_channels=num_kp, 
            kernel_size=3, padding=1)

        self.temperature = temperature
        self.scale_factor = scale_factor
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(image_channel, self.scale_factor)
        else: # To support torch.jit.script
            self.down = nn.Identity()

        # load_weights(self, "models/KPDetector.pth")

        # torch.jit.script(self)  ==> Error
        # torch.jit.script(self.predictor) ==> Error
        # torch.jit.script(self.kp)
        # torch.jit.script(self.down)

    def gaussian2kp(self, heatmap) -> Dict[str, torch.Tensor]:
        """
        Extract the mean from a heatmap
        """
        shape = heatmap.shape
        heatmap = heatmap.unsqueeze(-1)
        grid = make_coordinate_grid(shape[2:], heatmap.type()).unsqueeze_(0).unsqueeze_(0)
        value = (heatmap * grid).sum(dim=(2, 3, 4))

        return value

    def forward(self, x):
        x = self.down(x)

        feature_map = self.predictor(x)
        prediction = self.kp(feature_map)

        final_shape = prediction.shape
        heatmap = prediction.view(final_shape[0], final_shape[1], -1)
        heatmap = F.softmax(heatmap / self.temperature, dim=2)
        heatmap = heatmap.view(*final_shape)

        out = self.gaussian2kp(heatmap)

        return out


class AntiAliasInterpolation2d(nn.Module):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """
    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ],
            indexing = "ij",
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale
        inv_scale = 1 / scale
        self.int_inv_scale = int(inv_scale)

    def forward(self, input):
        if self.scale == 1.0:
            return input

        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = out[:, :, ::self.int_inv_scale, ::self.int_inv_scale]

        return out


class KPHourglass(nn.Module):
    """
    KPHourglass architecture.
    """ 

    def __init__(self, block_expansion, in_features, reshape_features, reshape_depth, num_blocks=3, max_features=256):
        super(KPHourglass, self).__init__()
        
        self.down_blocks = nn.Sequential()
        for i in range(num_blocks):
            self.down_blocks.add_module('down'+ str(i), 
                DownBlock2d(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                           min(max_features, block_expansion * (2 ** (i + 1))),
                           kernel_size=3, padding=1))

        in_filters = min(max_features, block_expansion * (2 ** num_blocks))
        self.conv = nn.Conv2d(in_channels=in_filters, out_channels=reshape_features, kernel_size=1)

        self.up_blocks = nn.Sequential()
        for i in range(num_blocks):
            in_filters = min(max_features, block_expansion * (2 ** (num_blocks - i)))
            out_filters = min(max_features, block_expansion * (2 ** (num_blocks - i - 1)))
            self.up_blocks.add_module('up'+ str(i), UpBlock3d(in_filters, out_filters, kernel_size=3, padding=1))

        self.reshape_depth = reshape_depth
        self.out_filters = out_filters

    def forward(self, x):
        out = self.down_blocks(x)
        out = self.conv(out)
        bs, c, h, w = out.shape
        out = out.view(bs, c//self.reshape_depth, self.reshape_depth, h, w)
        out = self.up_blocks(out)

        return out
        
if __name__ == "__main__":
    model = KPDetector()
    print(model)