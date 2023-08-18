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
from torch.nn.utils import spectral_norm, remove_spectral_norm
from SAD.dense_motion import DenseMotionNetwork
from SAD.util import load_weights, DownBlock2d

from typing import Dict
import pdb

def remove_sadkernel_spectral_norm(model):
    remove_spectral_norm(model.decoder.G_middle_0.conv_0)
    remove_spectral_norm(model.decoder.G_middle_0.conv_1)
    remove_spectral_norm(model.decoder.G_middle_1.conv_0)
    remove_spectral_norm(model.decoder.G_middle_1.conv_1)
    remove_spectral_norm(model.decoder.G_middle_2.conv_0)
    remove_spectral_norm(model.decoder.G_middle_2.conv_1)
    remove_spectral_norm(model.decoder.G_middle_3.conv_0)
    remove_spectral_norm(model.decoder.G_middle_3.conv_1)
    remove_spectral_norm(model.decoder.G_middle_4.conv_0)
    remove_spectral_norm(model.decoder.G_middle_4.conv_1)
    remove_spectral_norm(model.decoder.G_middle_5.conv_0)
    remove_spectral_norm(model.decoder.G_middle_5.conv_1)

    remove_spectral_norm(model.decoder.up_0.conv_0)
    remove_spectral_norm(model.decoder.up_0.conv_1)
    remove_spectral_norm(model.decoder.up_0.conv_s)
    remove_spectral_norm(model.decoder.up_1.conv_0)
    remove_spectral_norm(model.decoder.up_1.conv_1)
    remove_spectral_norm(model.decoder.up_1.conv_s)    


# comes from "first order motion" ?
class SADKernel(nn.Module):
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
        super(SADKernel, self).__init__()
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

        # load_weights(self, "models/SADKernel.pth")

        # torch.jit.script(self) ==> Error !
        # torch.jit.script(self.decoder) ==> AttributeError: 'SpectralNorm' object has no attribute '__name__'
        # torch.jit.script(self.third)
        # torch.jit.script(self.fourth)


    def deform_input(self, inp, deformation):
        _, d_old, h_old, w_old, _ = deformation.shape
        _, _, d, h, w = inp.shape
        if d_old != d or h_old != h or w_old != w:
            deformation = deformation.permute(0, 4, 1, 2, 3)
            deformation = F.interpolate(deformation, size=(d, h, w), mode='trilinear')
            deformation = deformation.permute(0, 2, 3, 4, 1)
        return F.grid_sample(inp, deformation, align_corners=False)

    def forward(self, source_image, audio_kp, image_kp):
        # Encoding (downsampling) part
        out = self.first(source_image)

        # for i in range(len(self.down_blocks)):
        #     out = self.down_blocks[i](out)
        for i, m in enumerate(self.down_blocks):
            out = m(out)

        out = self.second(out)
        bs, c, h, w = out.shape
        # print(out.shape)
        feature_3d = out.view(bs, self.reshape_channel, self.reshape_depth, h ,w) 
        feature_3d = self.resblocks_3d(feature_3d)

        # Transforming feature representation according to deformation and occlusion
        output_dict = {}
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # feature_3d.size() -- [2, 32, 16, 64, 64]
        # dense_motion -- DenseMotionNetwork(...)
        dense_motion = self.dense_motion_network(feature=feature_3d, audio_kp=audio_kp,
                                                 image_kp=image_kp)

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
        return out # size() -- [2, 3, 256, 256]

class SPADEDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        ic = 256
        oc = 64
        label_nc = 256
        norm_G = 'spadespectralinstance'
        
        self.fc = nn.Conv2d(ic, 2 * ic, 3, padding=1)
        self.G_middle_0 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_1 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_2 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_3 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_4 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_5 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.up_0 = ShortcutSPADEResnetBlock(2 * ic, ic, norm_G, label_nc)
        self.up_1 = ShortcutSPADEResnetBlock(ic, oc, norm_G, label_nc)
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
        x = F.sigmoid(x)
        
        return x

class SameBlock2d(nn.Module):
    """
    Simple block, preserve spatial resolution.
    """

    def __init__(self, in_features, out_features, groups=1, kernel_size=3, padding=1, lrelu=False):
        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = nn.BatchNorm2d(out_features, affine=True)
        if lrelu:
            self.ac = nn.LeakyReLU()
        else:
            self.ac = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.ac(out)
        return out


class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()
        nhidden = 128
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU())
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):
        normalized = self.param_free_norm(x)
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma) + beta
        return out
    

class SPADEResnetBlock(nn.Module):
    '''
        self.learned_shortcut = (fin != fout),  False
    '''    
    def __init__(self, fin, fout, norm_G, label_nc, dilation=1):
        super().__init__()
        # norm_G === 'spadespectralinstance'

        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=dilation, dilation=dilation)

        # apply spectral norm if specified, xxxx8888
        self.conv_0 = spectral_norm(self.conv_0)
        self.conv_1 = spectral_norm(self.conv_1)

        # define normalization layers
        self.norm_0 = SPADE(fin, label_nc)
        self.norm_1 = SPADE(fmiddle, label_nc)

    def forward(self, x, seg1):
        x_s = self.shortcut(x, seg1)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg1)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg1)))
        out = x_s + dx
        return out

    def shortcut(self, x, seg1):
        return x

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class ShortcutSPADEResnetBlock(nn.Module):
    '''
        self.learned_shortcut = (fin != fout),  True
    '''
    def __init__(self, fin, fout, norm_G, label_nc, dilation=1):
        super().__init__()
        # norm_G === 'spadespectralinstance'

        # Attributes
        fmiddle = min(fin, fout)
        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        # xxxx8888
        self.conv_0 = spectral_norm(self.conv_0)
        self.conv_1 = spectral_norm(self.conv_1)
        self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        self.norm_0 = SPADE(fin, label_nc)
        self.norm_1 = SPADE(fmiddle, label_nc)
        self.norm_s = SPADE(fin, label_nc)

    def forward(self, x, seg1):
        x_s = self.shortcut(x, seg1)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg1)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg1)))
        out = x_s + dx
        return out

    def shortcut(self, x, seg1):
        return self.conv_s(self.norm_s(x, seg1))

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class ResBlock3d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock3d, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv3d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.norm1 = nn.BatchNorm3d(in_features, affine=True)
        self.norm2 = nn.BatchNorm3d(in_features, affine=True)

    def forward(self, x):
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out

if __name__ == "__main__":
    model = SADKernel()

    remove_sadkernel_spectral_norm(model)
    model = torch.jit.script(model)    
    print(model)
