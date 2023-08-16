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
import torch.nn as nn
import torch.nn.functional as F
from SAD.util import make_coordinate_grid, UpBlock3d
from typing import Dict
import pdb

class DenseMotionNetwork(nn.Module):
    """
    Module that predicting a dense motion from sparse motion representation given by image_kp and audio_kp
    src/config/facerender.yaml
        dense_motion_params:
          block_expansion: 32
          max_features: 1024
          num_blocks: 5
          reshape_depth: 16
          compress: 4    
    """

    def __init__(self, 
                block_expansion=32, 
                num_blocks=5, 
                max_features=1024, 
                num_kp=15, 
                feature_channel=32, 
                reshape_depth=16, 
                compress=4,
            ):
        super(DenseMotionNetwork, self).__init__()
        self.hourglass = Hourglass(
                block_expansion=block_expansion, 
                in_features=(num_kp+1)*(compress+1), 
                max_features=max_features, 
                num_blocks=num_blocks)

        self.mask = nn.Conv3d(self.hourglass.out_filters, num_kp + 1, kernel_size=7, padding=3)

        self.compress = nn.Conv3d(feature_channel, compress, kernel_size=1)
        self.norm = nn.BatchNorm3d(compress, affine=True)
        self.occlusion = nn.Conv2d(self.hourglass.out_filters*reshape_depth, 1, kernel_size=7, padding=3)
        self.num_kp = num_kp


    def create_sparse_motions(self, feature, audio_kp, image_kp):
        bs, _, d, h, w = feature.shape
        identity_grid = make_coordinate_grid((d, h, w), type=image_kp.type())
        identity_grid = identity_grid.view(1, 1, d, h, w, 3)
        coordinate_grid = identity_grid - audio_kp.view(bs, self.num_kp, 1, 1, 1, 3)
        
        # # if 'jacobian' in audio_kp:
        # if 'jacobian' in audio_kp and audio_kp['jacobian'] is not None:
        #     jacobian = torch.matmul(image_kp['jacobian'], torch.inverse(audio_kp['jacobian']))
        #     jacobian = jacobian.unsqueeze(-3).unsqueeze(-3).unsqueeze(-3)
        #     jacobian = jacobian.repeat(1, 1, d, h, w, 1, 1)
        #     coordinate_grid = torch.matmul(jacobian, coordinate_grid.unsqueeze(-1))
        #     coordinate_grid = coordinate_grid.squeeze(-1)                  


        driving_to_source = coordinate_grid + image_kp.view(bs, self.num_kp, 1, 1, 1, 3)    # (bs, num_kp, d, h, w, 3)

        #adding background feature
        identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1, 1)
        sparse_motions = torch.cat([identity_grid, driving_to_source], dim=1)                #bs num_kp+1 d h w 3
        
        # sparse_motions = driving_to_source

        return sparse_motions

    def create_deformed_feature(self, feature, sparse_motions):
        bs, _, d, h, w = feature.shape
        feature_repeat = feature.unsqueeze(1).unsqueeze(1).repeat(1, self.num_kp+1, 1, 1, 1, 1, 1)      # (bs, num_kp+1, 1, c, d, h, w)
        feature_repeat = feature_repeat.view(bs * (self.num_kp+1), -1, d, h, w)                         # (bs*(num_kp+1), c, d, h, w)
        sparse_motions = sparse_motions.view((bs * (self.num_kp+1), d, h, w, -1))                       # (bs*(num_kp+1), d, h, w, 3) !!!!
        sparse_deformed = F.grid_sample(feature_repeat, sparse_motions, align_corners=False)
        sparse_deformed = sparse_deformed.view((bs, self.num_kp+1, -1, d, h, w))                        # (bs, num_kp+1, c, d, h, w)
        return sparse_deformed

    def create_heatmap_representations(self, feature, audio_kp, image_kp):
        spatial_size = feature.shape[3:]
        gaussian_driving = kp2gaussian(audio_kp, spatial_size=spatial_size, kp_variance=0.01)
        gaussian_source = kp2gaussian(image_kp, spatial_size=spatial_size, kp_variance=0.01)
        heatmap = gaussian_driving - gaussian_source

        # adding background feature
        zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1], spatial_size[2]).type(heatmap.type())
        heatmap = torch.cat([zeros, heatmap], dim=1)
        heatmap = heatmap.unsqueeze(2)         # (bs, num_kp+1, 1, d, h, w)
        return heatmap

    def forward(self, feature, audio_kp, image_kp) -> Dict[str, torch.Tensor]:
        bs, _, d, h, w = feature.shape

        feature = self.compress(feature)
        feature = self.norm(feature)
        feature = F.relu(feature)

        out_dict = dict()
        sparse_motion = self.create_sparse_motions(feature, audio_kp, image_kp)
        deformed_feature = self.create_deformed_feature(feature, sparse_motion)

        heatmap = self.create_heatmap_representations(deformed_feature, audio_kp, image_kp)

        input_ = torch.cat([heatmap, deformed_feature], dim=2)
        input_ = input_.view(bs, -1, d, h, w)

        # input = deformed_feature.view(bs, -1, d, h, w)      # (bs, num_kp+1 * c, d, h, w)

        prediction = self.hourglass(input_)


        mask = self.mask(prediction)
        mask = F.softmax(mask, dim=1)
        out_dict['mask'] = mask
        mask = mask.unsqueeze(2)                                   # (bs, num_kp+1, 1, d, h, w)
        
        zeros_mask = torch.zeros_like(mask)   
        mask = torch.where(mask < 1e-3, zeros_mask, mask) 

        sparse_motion = sparse_motion.permute(0, 1, 5, 2, 3, 4)    # (bs, num_kp+1, 3, d, h, w)
        deformation = (sparse_motion * mask).sum(dim=1)            # (bs, 3, d, h, w)
        deformation = deformation.permute(0, 2, 3, 4, 1)           # (bs, d, h, w, 3)

        out_dict['deformation'] = deformation

        bs, c, d, h, w = prediction.shape
        prediction = prediction.view(bs, -1, h, w)
        occlusion_map = torch.sigmoid(self.occlusion(prediction))
        out_dict['occlusion_map'] = occlusion_map

        return out_dict

class Encoder(nn.Module):
    """
    Hourglass Encoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Encoder, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock3d(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                           min(max_features, block_expansion * (2 ** (i + 1))),
                                           kernel_size=3, padding=1))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs


class Decoder(nn.Module):
    """
    Hourglass Decoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Decoder, self).__init__()

        up_blocks = []

        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 1)))
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(UpBlock3d(in_filters, out_filters, kernel_size=3, padding=1))

        self.up_blocks = nn.ModuleList(up_blocks)
        # self.out_filters = block_expansion
        self.out_filters = block_expansion + in_features

        self.conv = nn.Conv3d(in_channels=self.out_filters, out_channels=self.out_filters, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm3d(self.out_filters, affine=True)

    def forward(self, x):
        out = x.pop()
        # for up_block in self.up_blocks[:-1]:
        for up_block in self.up_blocks:
            out = up_block(out)
            skip = x.pop()
            out = torch.cat([out, skip], dim=1)
        # out = self.up_blocks[-1](out)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out

class Hourglass(nn.Module):
    """
    Hourglass architecture.
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Hourglass, self).__init__()
        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder(block_expansion, in_features, num_blocks, max_features)
        self.out_filters = self.decoder.out_filters

    def forward(self, x):
        return self.decoder(self.encoder(x))

def kp2gaussian(kp, spatial_size, kp_variance):
    """
    Transform a keypoint into gaussian like representation
    """
    mean = kp

    coordinate_grid = make_coordinate_grid(spatial_size, mean.type())
    number_of_leading_dimensions = len(mean.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    coordinate_grid = coordinate_grid.view(*shape)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)

    # Preprocess kp shape
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 1, 3)
    mean = mean.view(*shape)

    mean_sub = (coordinate_grid - mean)

    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)

    return out

class DownBlock3d(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock3d, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = nn.BatchNorm3d(out_features, affine=True)
        self.pool = nn.AvgPool3d(kernel_size=(1, 2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out

