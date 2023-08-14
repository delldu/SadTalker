import torch
from torch import nn
import torch.nn.functional as F
from src.facerender.modules.util import SameBlock2d, DownBlock2d, ResBlock3d, SPADEResnetBlock
from src.facerender.modules.dense_motion import DenseMotionNetwork
import pdb


class SPADEDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        ic = 256
        oc = 64
        norm_G = 'spadespectralinstance'
        label_nc = 256
        
        self.fc = nn.Conv2d(ic, 2 * ic, 3, padding=1)
        self.G_middle_0 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_1 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_2 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_3 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_4 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_5 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.up_0 = SPADEResnetBlock(2 * ic, ic, norm_G, label_nc)
        self.up_1 = SPADEResnetBlock(ic, oc, norm_G, label_nc)
        self.conv_img = nn.Conv2d(oc, 3, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2)
        
    def forward(self, feature):
        seg = feature
        x = self.fc(feature)
        x = self.G_middle_0(x, seg)
        x = self.G_middle_1(x, seg)
        x = self.G_middle_2(x, seg)
        x = self.G_middle_3(x, seg)
        x = self.G_middle_4(x, seg)
        x = self.G_middle_5(x, seg)
        x = self.up(x)                
        x = self.up_0(x, seg)         # 256, 128, 128
        x = self.up(x)                
        x = self.up_1(x, seg)         # 64, 256, 256

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        # x = torch.tanh(x)
        x = F.sigmoid(x)
        
        return x


class OcclusionAwareSPADEGenerator(nn.Module):
    '''
    src/config/facerender.yaml
      generator_params:
        block_expansion: 64
        max_features: 512
        num_down_blocks: 2
        reshape_channel: 32
        reshape_depth: 16         # 512 = 32 * 16
        num_resblocks: 6
        estimate_occlusion_map: True
    '''

    def __init__(self, 
                 image_channel=3 , 
                 feature_channel=32, 
                 num_kp=15, 
                 block_expansion=64, 
                 max_features=512, 
                 num_down_blocks=2, 
                 reshape_channel=32, 
                 reshape_depth=16, 
                 num_resblocks=6,
                 # estimate_occlusion_map=True,
                 # dense_motion_params=None,
                 # estimate_jacobian=False, 
                ):
        super(OcclusionAwareSPADEGenerator, self).__init__()
        self.dense_motion_network = DenseMotionNetwork(num_kp=num_kp, feature_channel=feature_channel)

        self.first = SameBlock2d(image_channel, block_expansion, kernel_size=(3, 3), padding=(1, 1))

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        self.second = nn.Conv2d(in_channels=out_features, out_channels=max_features, kernel_size=1, stride=1)

        self.reshape_channel = reshape_channel
        self.reshape_depth = reshape_depth

        self.resblocks_3d = torch.nn.Sequential()
        for i in range(num_resblocks):
            self.resblocks_3d.add_module('3dr' + str(i), ResBlock3d(reshape_channel, kernel_size=3, padding=1))

        out_features = block_expansion * (2 ** (num_down_blocks))
        self.third = SameBlock2d(max_features, out_features, kernel_size=(3, 3), padding=(1, 1), lrelu=True)
        self.fourth = nn.Conv2d(in_channels=out_features, out_channels=out_features, kernel_size=1, stride=1)

        self.decoder = SPADEDecoder()

    def deform_input(self, inp, deformation):
        _, d_old, h_old, w_old, _ = deformation.shape
        _, _, d, h, w = inp.shape
        if d_old != d or h_old != h or w_old != w:
            deformation = deformation.permute(0, 4, 1, 2, 3)
            deformation = F.interpolate(deformation, size=(d, h, w), mode='trilinear')
            deformation = deformation.permute(0, 2, 3, 4, 1)
        return F.grid_sample(inp, deformation, align_corners=False)

    def forward(self, source_image, kp_driving, kp_source):
        # Encoding (downsampling) part
        out = self.first(source_image)
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
        out = self.second(out)
        bs, c, h, w = out.shape
        # print(out.shape)
        feature_3d = out.view(bs, self.reshape_channel, self.reshape_depth, h ,w) 
        feature_3d = self.resblocks_3d(feature_3d)

        # Transforming feature representation according to deformation and occlusion
        output_dict = {}

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # feature_3d.size() -- [2, 32, 16, 64, 64]
        dense_motion = self.dense_motion_network(feature=feature_3d, kp_driving=kp_driving,
                                                 kp_source=kp_source)

        # dense_motion.keys() -- ['mask', 'deformation', 'occlusion_map']
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        output_dict['mask'] = dense_motion['mask']
        occlusion_map = dense_motion['occlusion_map']
        output_dict['occlusion_map'] = occlusion_map
        deformation = dense_motion['deformation'] # size() -- [2, 16, 64, 64, 3]

        out = self.deform_input(feature_3d, deformation)

        bs, c, d, h, w = out.shape
        out = out.view(bs, c*d, h, w)
        out = self.third(out)
        out = self.fourth(out)

        if out.shape[2] != occlusion_map.shape[2] or out.shape[3] != occlusion_map.shape[3]:
            occlusion_map = F.interpolate(occlusion_map, size=out.shape[2:], mode='bilinear')
        out = out * occlusion_map

        # Decoding part
        out = self.decoder(out)

        output_dict["prediction"] = out # size() -- [2, 3, 256, 256]

        return output_dict