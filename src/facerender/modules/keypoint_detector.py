import torch
from torch import nn
import torch.nn.functional as F

from src.facerender.modules.util import (
    KPHourglass, 
    make_coordinate_grid, 
)
from typing import Dict

import pdb

class KPDetector(nn.Module):
    """
    Detecting canonical keypoints. Return keypoint position and jacobian near each keypoint.

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
        kp = {'value': value}

        return kp

    def forward(self, x) -> Dict[str, torch.Tensor]:
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

