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
from typing import List, Tuple
import todos
import pdb

class DenseMotionNetwork(nn.Module):
    """
    Predict a dense motion from sparse motion representation 
    given by image_kp and audio_kp
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
        super().__init__()
        self.hourglass = Hourglass(
                block_expansion=block_expansion, 
                in_features=(num_kp+1)*(compress+1), 
                max_features=max_features, 
                num_blocks=num_blocks)

        self.mask = nn.Conv3d(self.hourglass.out_filters, num_kp + 1, kernel_size=7, padding=3)

        self.compress = nn.Conv3d(feature_channel, compress, kernel_size=1)
        self.norm = nn.BatchNorm3d(compress) # , affine=True)
        self.occlusion = nn.Conv2d(self.hourglass.out_filters*reshape_depth, 1, kernel_size=7, padding=3)
        self.num_kp = num_kp


    def create_sparse_motions(self, feature, audio_kp, image_kp):
        bs, _, d, h, w = feature.shape
        identity_grid = make_coordinate_grid((d, h, w)).to(feature.device)
        identity_grid = identity_grid.view(1, 1, d, h, w, 3)
        coordinate_grid = identity_grid - audio_kp.view(bs, self.num_kp, 1, 1, 1, 3)
        
        driving_to_source = coordinate_grid + image_kp.view(bs, self.num_kp, 1, 1, 1, 3)

        #Add background feature
        identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1, 1)
        sparse_motions = torch.cat([identity_grid, driving_to_source], dim=1)

        return sparse_motions

    def create_deformed_feature(self, feature, sparse_motions):
        # tensor [feature] size: [1, 4, 16, 128, 128], min: 0.0, max: 2.637389, mean: 0.091283
        # tensor [sparse_motions] size: [1, 16, 16, 128, 128, 3], min: -1.089367, max: 1.029755, mean: -0.020543

        bs, _, d, h, w = feature.shape
        feature_repeat = feature.unsqueeze(1).unsqueeze(1).repeat(1, self.num_kp+1, 1, 1, 1, 1, 1)
        feature_repeat = feature_repeat.view(bs * (self.num_kp+1), -1, d, h, w)
        sparse_motions = sparse_motions.view((bs * (self.num_kp+1), d, h, w, -1))

        # tensor [feature_repeat] size: [16, 4, 16, 128, 128], min: 0.0, max: 2.637389, mean: 0.091283
        # tensor [sparse_motions] size: [16, 16, 128, 128, 3], min: -1.089367, max: 1.029755, mean: -0.020543
        sparse_deformed = F.grid_sample(feature_repeat, sparse_motions, align_corners=False)
        # tensor [sparse_deformed] size: [16, 4, 16, 128, 128], min: 0.0, max: 2.138518, mean: 0.080648

        sparse_deformed = sparse_deformed.view((bs, self.num_kp+1, -1, d, h, w))
        return sparse_deformed

    def create_heat_map(self, deform_feature, audio_kp, image_kp):
        # tensor [deform_feature] size: [1, 16, 4, 16, 128, 128], min: 0.0, max: 2.1357, mean: 0.080596
        # tensor [audio_kp] size: [1, 15, 3], min: -1.062631, max: 0.829658, mean: -0.0615
        # tensor [image_kp] size: [1, 15, 3], min: -1.141619, max: 0.832342, mean: -0.083755

        spatial_size = (deform_feature.shape[3], deform_feature.shape[4], deform_feature.shape[5])
        # spatial_size -- [16, 128, 128]

        gaussian_audio = keypoint2gaussion(audio_kp, spatial_size=spatial_size)
        gaussian_image = keypoint2gaussion(image_kp, spatial_size=spatial_size)
        heatmap = gaussian_audio - gaussian_image

        # Adding background feature
        zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], 
            spatial_size[1], spatial_size[2]).type(heatmap.type())
        heatmap = torch.cat([zeros, heatmap], dim=1)
        heatmap = heatmap.unsqueeze(2)

        return heatmap # size() -- [1, 16, 1, 16, 128, 128]

    def forward(self, feature, audio_kp, image_kp) -> Tuple[torch.Tensor, torch.Tensor]:
        # tensor [feature] size: [1, 32, 16, 128, 128], min: -38.411057, max: 36.868614, mean: 1.234772
        # tensor [audio_kp] size: [1, 15, 3], min: -0.983468, max: 0.936596, mean: -0.00567
        # tensor [image_kp] size: [1, 15, 3], min: -0.847928, max: 0.93429, mean: 0.040402
        
        bs, _, d, h, w = feature.shape

        feature = self.compress(feature)
        feature = self.norm(feature)
        feature = F.relu(feature)

        sparse_motion = self.create_sparse_motions(feature, audio_kp, image_kp)
        deformed_feature = self.create_deformed_feature(feature, sparse_motion)

        heatmap = self.create_heat_map(deformed_feature, audio_kp, image_kp)

        input_ = torch.cat([heatmap, deformed_feature], dim=2)
        input_ = input_.view(bs, -1, d, h, w)
        prediction = self.hourglass(input_)

        mask = self.mask(prediction)
        mask = F.softmax(mask, dim=1)
        mask = mask.unsqueeze(2) # size(): [1, 16, 16, 128, 128] ==> [1, 16, 1, 16, 128, 128]
        # zeros_mask = torch.zeros_like(mask)   
        # mask = torch.where(mask < 1e-3, zeros_mask, mask) 
        mask = mask.clamp(0.0)

        sparse_motion = sparse_motion.permute(0, 1, 5, 2, 3, 4) # size() -- [1, 16, 3, 16, 128, 128]
        deformation = (sparse_motion * mask).sum(dim=1)
        deformation = deformation.permute(0, 2, 3, 4, 1)

        bs, c, d, h, w = prediction.shape
        prediction = prediction.view(bs, -1, h, w)
        occlusion_map = torch.sigmoid(self.occlusion(prediction))
        # tensor [deformation] size: [1, 16, 128, 128, 3], min: -1.006518, max: 1.07888, mean: 0.028499
        # tensor [occlusion_map] size: [1, 1, 128, 128], min: 0.87407, max: 0.998679, mean: 0.967596
        return (deformation, occlusion_map)

class HourglassEncoder(nn.Module):
    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super().__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock3d(
                in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                min(max_features, block_expansion * (2 ** (i + 1))),
                kernel_size=3, padding=1))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        # tensor [x] size: [1, 80, 16, 128, 128], min: -0.74504, max: 1.718306, mean: 0.059655
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))

        # outs is list: len = 6
        #     tensor [item] size: [1, 80, 16, 128, 128], min: -0.74504, max: 1.718306, mean: 0.059655
        #     tensor [item] size: [1, 64, 16, 64, 64], min: 0.0, max: 13.028785, mean: 0.24038
        #     tensor [item] size: [1, 128, 16, 32, 32], min: 0.0, max: 14.465733, mean: 0.17052
        #     tensor [item] size: [1, 256, 16, 16, 16], min: 0.0, max: 9.857283, mean: 0.15933
        #     tensor [item] size: [1, 512, 16, 8, 8], min: 0.0, max: 4.83479, mean: 0.163259
        #     tensor [item] size: [1, 1024, 16, 4, 4], min: 0.0, max: 4.537486, mean: 0.163623
        return outs


class HourglassDecoder(nn.Module):
    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super().__init__()

        up_blocks = []

        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 1)))
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(UpBlock3d(in_filters, out_filters, kernel_size=3, padding=1))

        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_filters = block_expansion + in_features

        self.conv = nn.Conv3d(in_channels=self.out_filters, out_channels=self.out_filters, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm3d(self.out_filters) #, affine=True)

    def forward(self, x: List[torch.Tensor]):
        # x is list: len = 6
        #     tensor [item] size: [1, 80, 16, 128, 128], min: -0.735851, max: 1.619775, mean: 0.059744
        #     tensor [item] size: [1, 64, 16, 64, 64], min: 0.0, max: 12.833364, mean: 0.240246
        #     tensor [item] size: [1, 128, 16, 32, 32], min: 0.0, max: 14.146493, mean: 0.170355
        #     tensor [item] size: [1, 256, 16, 16, 16], min: 0.0, max: 9.610794, mean: 0.15924
        #     tensor [item] size: [1, 512, 16, 8, 8], min: 0.0, max: 4.751517, mean: 0.163225
        #     tensor [item] size: [1, 1024, 16, 4, 4], min: 0.0, max: 4.511559, mean: 0.163519
        out = x.pop()
        for up_block in self.up_blocks:
            out = up_block(out)
            skip = x.pop()
            out = torch.cat([out, skip], dim=1)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        # tensor [out] size: [1, 112, 16, 128, 128], min: 0.0, max: 0.309183, mean: 0.004572
        return out

class Hourglass(nn.Module):
    """Hourglass architecture. """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super().__init__()
        self.encoder = HourglassEncoder(block_expansion, in_features, num_blocks, max_features)
        self.decoder = HourglassDecoder(block_expansion, in_features, num_blocks, max_features)
        self.out_filters = self.decoder.out_filters

    def forward(self, x):
        return self.decoder(self.encoder(x))

def keypoint2gaussion(kp, spatial_size: Tuple[int, int, int]):
    """
    Transform a keypoint into gaussian like representation
    """
    mean = kp # size() -- [1, 15, 3]
    # sparse_size -- (16, 128, 128)
    coordinate_grid = make_coordinate_grid(spatial_size).to(kp.device) # [16, 128, 128, 3]
    coordinate_grid = coordinate_grid.unsqueeze(0).unsqueeze(0) # [1, 1, 16, 128, 128, 3]
    coordinate_grid = coordinate_grid.repeat(1, 15, 1, 1, 1, 1)

    # Preprocess kp shape
    mean = mean.unsqueeze(2).unsqueeze(2).unsqueeze(2) # [1, 15, 3] ==> [1, 15, 1, 1, 1, 3]
    kp_variance = 0.01
    mean = (coordinate_grid - mean)
    out = torch.exp(-0.5 * (mean ** 2).sum(-1) / kp_variance)

    return out # size() -- [1, 15, 16, 128, 128]

class DownBlock3d(nn.Module):
    """
    Downsampling block for use in encoder.
    """
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, 
                        kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = nn.BatchNorm3d(out_features)
        self.pool = nn.AvgPool3d(kernel_size=(1, 2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out
