from torch import nn
import torch.nn.functional as F
import torch
from src.facerender.modules.util import Hourglass, make_coordinate_grid, kp2gaussian
from src.facerender.sync_batchnorm import SynchronizedBatchNorm3d as BatchNorm3d

from typing import Dict

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
        self.norm = BatchNorm3d(compress, affine=True)
        self.occlusion = nn.Conv2d(self.hourglass.out_filters*reshape_depth, 1, kernel_size=7, padding=3)
        self.num_kp = num_kp


    def create_sparse_motions(self, feature, audio_kp, image_kp):
        bs, _, d, h, w = feature.shape
        identity_grid = make_coordinate_grid((d, h, w), type=image_kp['value'].type())
        identity_grid = identity_grid.view(1, 1, d, h, w, 3)
        coordinate_grid = identity_grid - audio_kp['value'].view(bs, self.num_kp, 1, 1, 1, 3)
        
        # # if 'jacobian' in audio_kp:
        # if 'jacobian' in audio_kp and audio_kp['jacobian'] is not None:
        #     jacobian = torch.matmul(image_kp['jacobian'], torch.inverse(audio_kp['jacobian']))
        #     jacobian = jacobian.unsqueeze(-3).unsqueeze(-3).unsqueeze(-3)
        #     jacobian = jacobian.repeat(1, 1, d, h, w, 1, 1)
        #     coordinate_grid = torch.matmul(jacobian, coordinate_grid.unsqueeze(-1))
        #     coordinate_grid = coordinate_grid.squeeze(-1)                  


        driving_to_source = coordinate_grid + image_kp['value'].view(bs, self.num_kp, 1, 1, 1, 3)    # (bs, num_kp, d, h, w, 3)

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
