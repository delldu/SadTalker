import torch
from torch import nn
import torch.nn.functional as F

from src.facerender.modules.util import (
    KPHourglass, 
    make_coordinate_grid, 
    AntiAliasInterpolation2d, 
)
import pdb

class KeypointDetector(nn.Module):
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
        super(KeypointDetector, self).__init__()
        self.predictor = KPHourglass(block_expansion, in_features=image_channel,
                                     max_features=max_features, reshape_features=reshape_channel,
                                     reshape_depth=reshape_depth, num_blocks=num_blocks)

        self.kp = nn.Conv3d(in_channels=self.predictor.out_filters, out_channels=num_kp, 
            kernel_size=3, padding=1)

        self.temperature = temperature
        self.scale_factor = scale_factor
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(image_channel, self.scale_factor)

    def gaussian2kp(self, heatmap):
        """
        Extract the mean from a heatmap
        """
        shape = heatmap.shape
        heatmap = heatmap.unsqueeze(-1)
        grid = make_coordinate_grid(shape[2:], heatmap.type()).unsqueeze_(0).unsqueeze_(0)
        value = (heatmap * grid).sum(dim=(2, 3, 4))
        kp = {'value': value}

        return kp

    def forward(self, x):
        if self.scale_factor != 1:
            x = self.down(x)

        feature_map = self.predictor(x)
        prediction = self.kp(feature_map)

        final_shape = prediction.shape
        heatmap = prediction.view(final_shape[0], final_shape[1], -1)
        heatmap = F.softmax(heatmap / self.temperature, dim=2)
        heatmap = heatmap.view(*final_shape)

        out = self.gaussian2kp(heatmap)

        return out


