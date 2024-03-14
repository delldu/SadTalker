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
from SAD.util import load_weights

from typing import Dict, List
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
        self.audio_encoder.eval()
        for param in self.audio_encoder.parameters():
            param.requires_grad = False

        self.netG = CVAE()

        # load_weights(self, "models/Audio2Pose.pth")


    def forward(self, x: Dict[str, torch.Tensor]):
        # x is dict:
        #     tensor [audio_mels] size: [1, 200, 1, 80, 16], min: -4.0, max: 2.590095, mean: -1.017794
        #     tensor [image_exp_pose] size: [1, 200, 70], min: -1.156697, max: 1.459776, mean: 0.023419
        #     tensor [audio_ratio] size: [1, 200, 1], min: 0.0, max: 1.0, mean: 0.6565
        #     tensor [class] size: [1], min: 0.0, max: 0.0, mean: 0.0

        batch: Dict[str, torch.Tensor] = {}
        image_exp_pose = x['image_exp_pose'] # size() -- [1, 200, 70]
        batch['image_pose'] = x['image_exp_pose'][:, 0, -6:]  # [1, 200, 70] ==> [1, 6]
        batch['class'] = x['class'] # tensor([0], device='cuda:0')
        bs = image_exp_pose.shape[0]
        
        audio_mels= x['audio_mels'] # size() -- [1, 200, 1, 80, 16]
        indiv_mels_use = audio_mels[:, 1:] # size() -- [1, 199, 1, 80, 16], we regard the reference as the first frame
        num_frames = audio_mels.shape[1] # 200
        num_frames = int(num_frames) - 1 # ==> 199

        div = num_frames // self.seq_len
        re = num_frames % self.seq_len
        pose_pred_list: List[torch.Tensor] = [torch.zeros(batch['image_pose'].unsqueeze(1).shape, 
                                dtype=batch['image_pose'].dtype, 
                                device=batch['image_pose'].device)]

        for i in range(div):
            z = torch.randn(bs, self.latent_dim).to(image_exp_pose.device)
            batch['z'] = z
            audio_emb = self.audio_encoder(indiv_mels_use[:, i*self.seq_len:(i+1)*self.seq_len,:,:,:]) #bs seq_len 512
            batch['audio_emb'] = audio_emb

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            y = self.netG(batch) # CVAE(...)
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            pose_pred_list.append(y)  #list of bs seq_len 6
        
        if re != 0:
            z = torch.randn(bs, self.latent_dim).to(image_exp_pose.device)
            batch['z'] = z
            audio_emb = self.audio_encoder(indiv_mels_use[:, -1*self.seq_len:,:,:,:]) #bs seq_len  512
            if audio_emb.shape[1] != self.seq_len:
                pad_dim = self.seq_len-audio_emb.shape[1]
                pad_audio_emb = audio_emb[:, :1].repeat(1, pad_dim, 1) 
                audio_emb = torch.cat([pad_audio_emb, audio_emb], 1) 
            batch['audio_emb'] = audio_emb

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            y = self.netG(batch)  # CVAE(...)
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            pose_pred_list.append(y[:,-1*re:,:])   
        
        audio_pose_pred = torch.cat(pose_pred_list, dim = 1)
        pose_pred = image_exp_pose[:, :1, -6:] + audio_pose_pred  # [1, 200, 6]
        # tensor [pose_pred] size: [1, 200, 6], min: -0.786278, max: 0.285415, mean: -0.106551

        return pose_pred # [1, 200, 6]

class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
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

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

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
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        # Bridge
        x4 = self.bridge(x3)

        # Decode
        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)

        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)

        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)

        x10 = self.up_residual_conv3(x9)

        output = self.output_layer(x10)

        return output


class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)

class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

    def forward(self, audio_sequences):
        # tensor [audio_sequences] size: [1, 32, 1, 80, 16], min: -3.269791, max: 2.590095, mean: -0.766574
        B = audio_sequences.size(0)
        audio_sequences = torch.cat(
            [audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)

        audio_embedding = self.audio_encoder(audio_sequences)
        dim = audio_embedding.shape[1]
        audio_embedding = audio_embedding.reshape((B, -1, dim, 1, 1))
        # tensor [audio_embedding] size: [1, 32, 512, 1, 1], min: 0.0, max: 8.744249, mean: 0.4157

        output = audio_embedding.squeeze(-1).squeeze(-1)
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
        encoder_layer_sizes = [192, 128] # cfg.MODEL.CVAE.ENCODER_LAYER_SIZES
        decoder_layer_sizes = [128, 192] # cfg.MODEL.CVAE.DECODER_LAYER_SIZES
        latent_size = 64 # cfg.MODEL.CVAE.LATENT_SIZE
        num_classes = 46 # cfg.DATASET.NUM_CLASSES
        audio_emb_in_size = 512 # cfg.MODEL.CVAE.AUDIO_EMB_IN_SIZE
        audio_emb_out_size = 6 # cfg.MODEL.CVAE.AUDIO_EMB_OUT_SIZE
        seq_len = 32 # fg.MODEL.CVAE.SEQ_LEN

        self.latent_size = latent_size

        # self.encoder = Encoder(encoder_layer_sizes, latent_size, num_classes,
        #                         audio_emb_in_size, audio_emb_out_size, seq_len)
        self.decoder = Decoder(decoder_layer_sizes, latent_size, num_classes,
                                audio_emb_in_size, audio_emb_out_size, seq_len)
        # torch.jit.script(self) ==> Error, 210

    def forward(self, batch: Dict[str, torch.Tensor]):
        return self.decoder(batch)

class Decoder(nn.Module):
    def __init__(self, layer_sizes, latent_size, num_classes, 
                audio_emb_in_size, audio_emb_out_size, seq_len):
        super().__init__()

        self.resunet = ResUnet()
        # self.num_classes = num_classes # 46
        self.seq_len = seq_len

        self.MLP = nn.Sequential()
        input_size = latent_size + seq_len*audio_emb_out_size + 6
        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())
        
        self.pose_linear = nn.Linear(6, 6)
        self.linear_audio = nn.Linear(audio_emb_in_size, audio_emb_out_size)
        self.classbias = nn.Parameter(torch.randn(num_classes, latent_size)) # num_classes === 46

    def forward(self, batch: Dict[str, torch.Tensor]):
        # batch is dict:
        #     tensor [image_pose] size: [1, 6], min: -0.660511, max: 0.165923, mean: -0.111156
        #     tensor [class] size: [1], min: 0.0, max: 0.0, mean: 0.0
        #     tensor [z] size: [1, 64], min: -2.471357, max: 1.718462, mean: -0.13537
        #     tensor [audio_emb] size: [1, 32, 512], min: 0.0, max: 8.744249, mean: 0.4157

        z = batch['z']
        bs = z.shape[0] # 1
        class_id = batch['class'] # 0
        image_pose = batch['image_pose']
        audio_in = batch['audio_emb']

        audio_out = self.linear_audio(audio_in)
        audio_out = audio_out.reshape([bs, -1])
        class_bias = self.classbias[class_id]

        z = z + class_bias
        x_in = torch.cat([image_pose, z, audio_out], dim=-1)
        x_out = self.MLP(x_in)
        x_out = x_out.reshape((bs, self.seq_len, -1)) # self.seq_len === 32

        pose_emb = self.resunet(x_out.unsqueeze(1))
        y = self.pose_linear(pose_emb.squeeze(1))
        # tensor [y] size: [1, 32, 6], min: -0.131533, max: 0.050444, mean: -0.009271
        
        return y # size() -- [1, 32, 6]


if __name__ == "__main__":
    model = Audio2Pose()
    model = torch.jit.script(model)    
    print(model)
    # ==> OK
