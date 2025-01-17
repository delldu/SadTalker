import torch
from torch import nn
from torch.nn import functional as F
from src.utils.debug import debug_var
from typing import Dict
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
    def __init__(self, device='cuda'):
        super().__init__()
        self.seq_len = 32
        self.latent_dim = 64
        self.audio_encoder = AudioEncoder(device)
        self.audio_encoder.eval()
        for param in self.audio_encoder.parameters():
            param.requires_grad = False

        self.netG = CVAE()

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # debug_var("Audio2Pose.x", x)
        # Audio2Pose.x is dict:
        #     tensor audio_mels size: [1, 200, 1, 80, 16] , min: tensor(-4., device='cuda:0') , max: tensor(2.5998, device='cuda:0')
        #     tensor image_exp_pose size: [1, 200, 70] , min: tensor(-1.0968, device='cuda:0') , max: tensor(1.1307, device='cuda:0')
        ##    num_frames value: 200
        #     tensor audio_ratio size: [1, 200, 1] , min: tensor(0., device='cuda:0') , max: tensor(1., device='cuda:0')
        #     audio_name value: 'chinese_news'
        #     image_name value: 'dell'
        #     tensor class size: [1] , min: tensor(0, device='cuda:0') , max: tensor(0, device='cuda:0')

        batch = {}
        image_exp_pose = x['image_exp_pose'] # size() -- [1, 200, 70]
        batch['image_pose'] = x['image_exp_pose'][:, 0, -6:]  # [1, 200, 70] ==> [1, 6]
        batch['class'] = x['class'] # tensor([0], device='cuda:0')
        bs = image_exp_pose.shape[0]
        
        audio_mels= x['audio_mels'] # size() -- [1, 200, 1, 80, 16]
        indiv_mels_use = audio_mels[:, 1:] # size() -- [1, 199, 1, 80, 16], we regard the reference as the first frame
        num_frames = audio_mels.size(1) # x['num_frames'] # 200
        num_frames = int(num_frames) - 1 # ==> 199

        div = num_frames//self.seq_len
        re = num_frames%self.seq_len
        pose_motion_pred_list = [torch.zeros(batch['image_pose'].unsqueeze(1).shape, 
                                dtype=batch['image_pose'].dtype, 
                                device=batch['image_pose'].device)]

        for i in range(div):
            z = torch.randn(bs, self.latent_dim).to(image_exp_pose.device)
            batch['z'] = z
            audio_emb = self.audio_encoder(indiv_mels_use[:, i*self.seq_len:(i+1)*self.seq_len,:,:,:]) #bs seq_len 512
            batch['audio_emb'] = audio_emb

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            batch = self.netG(batch) # CVAE(...)
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            pose_motion_pred_list.append(batch['pose_motion_pred'])  #list of bs seq_len 6
        
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
            batch = self.netG(batch)  # CVAE(...)
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            pose_motion_pred_list.append(batch['pose_motion_pred'][:,-1*re:,:])   
        
        pose_motion_pred = torch.cat(pose_motion_pred_list, dim = 1)
        batch['pose_motion_pred'] = pose_motion_pred

        pose_pred = image_exp_pose[:, :1, -6:] + pose_motion_pred  # [1, 200, 6]

        batch['pose_pred'] = pose_pred # size() -- [1, 200, 6] 
        
        # debug_var("Audio2Pose.batch", batch)
        # Audio2Pose.batch is dict:
        # tensor image_pose size: [1, 6] , min: tensor(-0.0195, device='cuda:0') , max: tensor(0.2540, device='cuda:0')
        # tensor class size: [1] , min: tensor(0, device='cuda:0') , max: tensor(0, device='cuda:0')
        # tensor z size: [1, 64] , min: tensor(-1.8347, device='cuda:0') , max: tensor(1.8609, device='cuda:0')
        # tensor audio_emb size: [1, 32, 512] , min: tensor(0., device='cuda:0') , max: tensor(10.3144, device='cuda:0')
        # tensor pose_pred size: [1, 200, 6] , min: tensor(-0.0595, device='cuda:0') , max: tensor(0.5657, device='cuda:0')

        return batch

class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

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
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)

class ResUnet(nn.Module):
    def __init__(self, channel=1, filters=[32, 64, 128, 256]):
        super(ResUnet, self).__init__()

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
    def __init__(self, device):
        super(AudioEncoder, self).__init__()

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
        # audio_sequences = (B, T, 1, 80, 16)
        B = audio_sequences.size(0)

        audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)

        audio_embedding = self.audio_encoder(audio_sequences) # B, 512, 1, 1
        dim = audio_embedding.shape[1]
        audio_embedding = audio_embedding.reshape((B, -1, dim, 1, 1))

        return audio_embedding.squeeze(-1).squeeze(-1) #B seq_len+1 512 


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

    def forward(self, batch):
        '''
        class_id = batch['class']
        z = torch.randn([class_id.size(0), self.latent_size]).to(class_id.device)
        batch['z'] = z
        '''
        return self.decoder(batch)

class Decoder(nn.Module):
    def __init__(self, layer_sizes, latent_size, num_classes, 
                audio_emb_in_size, audio_emb_out_size, seq_len):
        super().__init__()

        self.resunet = ResUnet()
        self.num_classes = num_classes
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

        self.classbias = nn.Parameter(torch.randn(self.num_classes, latent_size))

    def forward(self, batch):
        # debug_var("Decoder.batch", batch)
        # Decoder.batch is dict:
        #     tensor image_pose size: [1, 6] , min: tensor(-0.0195, device='cuda:0') , max: tensor(0.2540, device='cuda:0')
        #     tensor class size: [1] , min: tensor(0, device='cuda:0') , max: tensor(0, device='cuda:0')
        #     tensor z size: [1, 64] , min: tensor(-2.7728, device='cuda:0') , max: tensor(1.7963, device='cuda:0')
        #     tensor audio_emb size: [1, 32, 512] , min: tensor(0., device='cuda:0') , max: tensor(8.4478, device='cuda:0')
        
        z = batch['z']                                          #bs latent_size
        bs = z.shape[0]
        class_id = batch['class']
        image_pose = batch['image_pose']                             #bs 6
        audio_in = batch['audio_emb']                           # bs seq_len audio_emb_in_size

        audio_out = self.linear_audio(audio_in)                 # bs seq_len audio_emb_out_size
        audio_out = audio_out.reshape([bs, -1])                 # bs seq_len*audio_emb_out_size
        class_bias = self.classbias[class_id]                   #bs latent_size

        z = z + class_bias
        x_in = torch.cat([image_pose, z, audio_out], dim=-1)
        x_out = self.MLP(x_in)                                  # bs layer_sizes[-1]
        x_out = x_out.reshape((bs, self.seq_len, -1))

        pose_emb = self.resunet(x_out.unsqueeze(1))             #bs 1 seq_len 6

        pose_motion_pred = self.pose_linear(pose_emb.squeeze(1))       #bs seq_len 6

        batch.update({'pose_motion_pred':pose_motion_pred}) # size() -- [1, 32, 6]

        # debug_var("Decoder.batch", batch)
        # Decoder.batch is dict:
        #     tensor image_pose size: [1, 6] , min: tensor(-0.0195, device='cuda:0') , max: tensor(0.2540, device='cuda:0')
        #     tensor class size: [1] , min: tensor(0, device='cuda:0') , max: tensor(0, device='cuda:0')
        #     tensor z size: [1, 64] , min: tensor(-2.7728, device='cuda:0') , max: tensor(1.7963, device='cuda:0')
        #     tensor audio_emb size: [1, 32, 512] , min: tensor(0., device='cuda:0') , max: tensor(8.4478, device='cuda:0')

        return batch


