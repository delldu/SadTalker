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
from typing import List
import todos
import pdb

class Audio2Pose(nn.Module):
    '''
    src/config/audio2pose.yaml

      CVAE:
        AUDIO_EMB_IN_SIZE: 512
        AUDIO_EMB_OUT_SIZE: 6
        SEQ_LEN: 32
        LATENT_SIZE: 64
        ENCODER_LAYER_SIZES: [192, 128]
        DECODER_LAYER_SIZES: [128, 192]
    '''
    def __init__(self):
        super().__init__()
        self.seq_len = 32
        self.latent_dim = 64
        self.audio_encoder = AudioEncoder()

        self.netG = CVAE()

    def forward(self, audio_mels, image_exp_pose, audio_ratio, class_id):
        # tensor [audio_mels] size: [1, 200, 80, 16], min: -4.0, max: 2.590095, mean: -1.017794
        # tensor [image_exp_pose] size: [1, 200, 70], min: -1.156697, max: 1.459776, mean: 0.023419
        # tensor [audio_ratio] size: [1, 200, 1], min: 0.0, max: 1.0, mean: 0.6565
        # tensor [class_id] size: [1], min: 0.0, max: 0.0, mean: 0.0

        B, C, H, W = audio_mels.size()
        pad = self.seq_len - (C % self.seq_len) + 1 # 25
        audio_mels_pad = torch.zeros(B, C + pad, H, W).to(audio_mels.device)
        audio_mels_pad[:, 0:C, :, :] = audio_mels
        # audio_mels_pad.size() -- [1, 225, 1, 80, 16]

        bs = image_exp_pose.shape[0]

        # image_pose = image_exp_pose[:, 0, -6:]  # [1, 200, 70] ==> [1, 6]
        image_pose = image_exp_pose[:, 0, 64:70]  # [1, 200, 70] ==> [1, 6]

        # we regard the reference as the first frame
        indiv_mels_use = audio_mels_pad[:, 1:] # size() -- [1, 224, 1, 80, 16], 
        num_frames = C + pad -1 # int(num_frames) - 1 # ==> 224
        div = num_frames // self.seq_len # ==> 7

        pose_pred_list: List[torch.Tensor] = [torch.zeros(image_pose.unsqueeze(1).shape, 
                                dtype=image_pose.dtype, 
                                device=image_pose.device)] # size() -- [1, 1, 6]
        for i in range(div):
            # Fix Bug: z randn make result is variable for onnx test, so we limit it with 0.05            
            # z = torch.randn(bs, self.latent_dim).to(image_exp_pose.device) # size() [1, 64]
            z = 0.05 * torch.randn(bs, self.latent_dim).to(image_exp_pose.device) # size() [1, 64]

            # indiv_mels_use -- [1, 224, 1, 80, 16]
            audio_emb = self.audio_encoder(indiv_mels_use[:, i*self.seq_len : (i+1)*self.seq_len, :, :])
            # audio_emb.size() -- [1, 32, 512]
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            y = self.netG(image_pose, class_id, z, audio_emb) # CVAE(...), size() -- [1, 32, 6]
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            pose_pred_list.append(y)
        
        audio_pose_pred = torch.cat(pose_pred_list, dim = 1) # size() -- [1, 200, 6]
        # image_exp_pose[:, :1, 64:70].size() -- [1, 1, 6]
        pose_pred = image_exp_pose[:, :1, 64:70] + audio_pose_pred  # [1, 200, 6]
        # tensor [pose_pred] size: [1, 200, 6], min: -0.786278, max: 0.285415, mean: -0.106551

        return pose_pred[:, 0:C, :] # [1, 200, 6]

class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(input_dim, output_dim, kernel_size=kernel, stride=stride)

    def forward(self, x):
        return self.upsample(x)

class ResUnet(nn.Module):
    def __init__(self, channel=1, filters=[32, 64, 128, 256]):
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], stride=(2,1), padding=1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], stride=(2,1), padding=1)

        self.bridge = ResidualConv(filters[2], filters[3], stride=(2,1), padding=1)

        self.upsample_1 = Upsample(filters[3], filters[3], kernel=(2,1), stride=(2,1))
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], stride=1, padding=1)

        self.upsample_2 = Upsample(filters[2], filters[2], kernel=(2,1), stride=(2,1))
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], stride=1, padding=1)

        self.upsample_3 = Upsample(filters[1], filters[1], kernel=(2,1), stride=(2,1))
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], stride=1, padding=1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], 1, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)

        # Bridge
        x4 = self.bridge(x3)

        # Decode
        x4 = self.upsample_1(x4)
        x6 = torch.cat([x4, x3], dim=1)

        x6 = self.up_residual_conv1(x6)
        x6 = self.upsample_2(x6)
        x8 = torch.cat([x6, x2], dim=1)
        x8 = self.up_residual_conv2(x8)
        x8 = self.upsample_3(x8)

        x9 = torch.cat([x8, x1], dim=1)
        x9 = self.up_residual_conv3(x9)

        output = self.output_layer(x9)

        return output.squeeze(1)


class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding):
        super().__init__()
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout),
                        )
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)

class Conv2dWithRes(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding):
        super().__init__()
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout),
                        )
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv_block(x)
        out += x # residual
        return self.act(out)        

class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2dWithRes(32, 32, kernel_size=3, stride=1, padding=1),
            Conv2dWithRes(32, 32, kernel_size=3, stride=1, padding=1),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2dWithRes(64, 64, kernel_size=3, stride=1, padding=1),
            Conv2dWithRes(64, 64, kernel_size=3, stride=1, padding=1),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2dWithRes(128, 128, kernel_size=3, stride=1, padding=1),
            Conv2dWithRes(128, 128, kernel_size=3, stride=1, padding=1),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2dWithRes(256, 256, kernel_size=3, stride=1, padding=1),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

    def forward(self, audio_sequences):
        # tensor [audio_sequences] size: [1, 32, 80, 16], min: -3.269791, max: 2.590095, mean: -0.766574
        B, C, H, W = audio_sequences.size()
        audio_sequences = audio_sequences.reshape(C, B, H, W)
        # size() ==> [32, 1, 80, 16]

        audio_embedding = self.audio_encoder(audio_sequences)
        dim = audio_embedding.shape[1]
        output = audio_embedding.reshape((B, -1, dim))
        # tensor [output] size: [1, 32, 512], min: 0.0, max: 8.744249, mean: 0.4157
        return output


class CVAE(nn.Module):
    '''
    src/config/audio2pose.yaml

    DATASET:
      NUM_CLASSES: 46

    MODEL:
      AUDIOENCODER:
        LEAKY_RELU: True
        NORM: 'IN'
      DISCRIMINATOR:
        LEAKY_RELU: False
        INPUT_CHANNELS: 6
      CVAE:
        AUDIO_EMB_IN_SIZE: 512
        AUDIO_EMB_OUT_SIZE: 6
        SEQ_LEN: 32
        LATENT_SIZE: 64
        ENCODER_LAYER_SIZES: [192, 128]
        DECODER_LAYER_SIZES: [128, 192]

    '''
    def __init__(self):
        super().__init__()
        # encoder_layer_sizes = [192, 128] # cfg.MODEL.CVAE.ENCODER_LAYER_SIZES
        decoder_layer_sizes = [128, 192] # cfg.MODEL.CVAE.DECODER_LAYER_SIZES
        latent_size = 64 # cfg.MODEL.CVAE.LATENT_SIZE
        num_classes = 46 # cfg.DATASET.NUM_CLASSES
        audio_emb_in_size = 512 # cfg.MODEL.CVAE.AUDIO_EMB_IN_SIZE
        audio_emb_out_size = 6 # cfg.MODEL.CVAE.AUDIO_EMB_OUT_SIZE
        seq_len = 32 # fg.MODEL.CVAE.SEQ_LEN
        self.decoder = Decoder(decoder_layer_sizes, latent_size, num_classes,
                                audio_emb_in_size, audio_emb_out_size, seq_len)

    def forward(self, image_pose, class_id, z, audio_emb):
        return self.decoder(image_pose, class_id, z, audio_emb)

class Decoder(nn.Module):
    def __init__(self, 
                 layer_sizes=[128, 192], 
                 latent_size=64, 
                 num_classes=46, 
                 audio_emb_in_size=512, 
                 audio_emb_out_size=6, 
                 seq_len=32,
            ):
        super().__init__()
        self.resunet = ResUnet()
        self.seq_len = seq_len

        self.MLP = nn.Sequential()
        input_size = latent_size + seq_len*audio_emb_out_size + 6 # ==> 262
        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            # [262, 128]
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())
        # self.MLP ---- 
        # Sequential(
        #   (L0): Linear(in_features=262, out_features=128, bias=True)
        #   (A0): ReLU()
        #   (L1): Linear(in_features=128, out_features=192, bias=True)
        #   (sigmoid): Sigmoid()
        # )
        self.pose_linear = nn.Linear(6, 6)
        self.linear_audio = nn.Linear(audio_emb_in_size, audio_emb_out_size)
        # self.linear_audio -- Linear(in_features=512, out_features=6, bias=True)
        self.classbias = nn.Parameter(torch.randn(num_classes, latent_size)) # size() -- [46, 64]

    def forward(self, image_pose, class_id, z, audio_emb):
        # tensor [image_pose] size: [1, 6], min: -0.660511, max: 0.165923, mean: -0.111156
        # tensor [class_id] size: [1], min: 0.0, max: 0.0, mean: 0.0
        # tensor [z] size: [1, 64], min: -2.471357, max: 1.718462, mean: -0.13537
        # tensor [audio_emb] size: [1, 32, 512], min: 0.0, max: 8.744249, mean: 0.4157
        bs = z.shape[0] # 1
        audio_out = self.linear_audio(audio_emb) # audio_emb.size() --  [1, 6]
        audio_out = audio_out.reshape([bs, -1]) # size() -- [1, 192]
        class_bias = self.classbias[class_id] # self.classbias.size() -- [46, 64]

        z = z + class_bias
        # tensor [image_pose] size: [1, 6], min: -2.225994, max: 0.265999, mean: -0.391909
        # tensor [z] size: [1, 64], min: -3.494162, max: 3.278047, mean: 0.214995
        # tensor [audio_out] size: [1, 192], min: -2.997055, max: 1.522663, mean: -0.083217
        x_in = torch.cat([image_pose, z, audio_out], dim=1) # size() -- [1, 262]
        x_out = self.MLP(x_in) # size() -- [1, 192]
        x_out = x_out.reshape((bs, self.seq_len, 6)) # self.seq_len === 32 ==> [1, 32, 6]

        pose_emb = self.resunet(x_out)
        y = self.pose_linear(pose_emb)
        # tensor [y] size: [1, 32, 6], min: -0.131533, max: 0.050444, mean: -0.009271
        
        return y # size() -- [1, 32, 6]
