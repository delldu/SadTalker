import torch
from torch import nn
from src.audio2pose_models.cvae import CVAE
from src.audio2pose_models.discriminator import PoseSequenceDiscriminator
from src.audio2pose_models.audio_encoder import AudioEncoder
from src.utils.debug import debug_var
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
        self.netD_motion = PoseSequenceDiscriminator() # useless ???

    def forward(self, x):
        # debug_var("Audio2Pose.x", x)
        # Audio2Pose.x is dict:
        #     tensor audio_mels size: [1, 200, 1, 80, 16] , min: tensor(-4., device='cuda:0') , max: tensor(2.5998, device='cuda:0')
        #     tensor image_exp_pose size: [1, 200, 70] , min: tensor(-1.0968, device='cuda:0') , max: tensor(1.1307, device='cuda:0')
        #     audio_num_frames value: 200
        #     tensor audio_ratio size: [1, 200, 1] , min: tensor(0., device='cuda:0') , max: tensor(1., device='cuda:0')
        #     audio_name value: 'chinese_news'
        #     image_name value: 'dell'
        #     tensor class size: [1] , min: tensor(0, device='cuda:0') , max: tensor(0, device='cuda:0')

        batch = {}
        image_exp_pose = x['image_exp_pose'] # size() -- [1, 200, 70]
        batch['pose'] = x['image_exp_pose'][:, 0, -6:]  # [1, 200, 70] ==> [1, 6]
        batch['class'] = x['class'] # tensor([0], device='cuda:0')
        bs = image_exp_pose.shape[0]
        
        audio_mels= x['audio_mels'] # size() -- [1, 200, 1, 80, 16]
        indiv_mels_use = audio_mels[:, 1:] # size() -- [1, 199, 1, 80, 16], we regard the reference as the first frame
        audio_num_frames = x['audio_num_frames'] # 200
        audio_num_frames = int(audio_num_frames) - 1 # ==> 199

        div = audio_num_frames//self.seq_len
        re = audio_num_frames%self.seq_len
        pose_motion_pred_list = [torch.zeros(batch['pose'].unsqueeze(1).shape, 
                                dtype=batch['pose'].dtype, 
                                device=batch['pose'].device)]

        for i in range(div):
            z = torch.randn(bs, self.latent_dim).to(image_exp_pose.device)
            batch['z'] = z
            audio_emb = self.audio_encoder(indiv_mels_use[:, i*self.seq_len:(i+1)*self.seq_len,:,:,:]) #bs seq_len 512
            batch['audio_emb'] = audio_emb

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            batch = self.netG(batch)
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
            batch = self.netG(batch)
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            pose_motion_pred_list.append(batch['pose_motion_pred'][:,-1*re:,:])   
        
        pose_motion_pred = torch.cat(pose_motion_pred_list, dim = 1)
        batch['pose_motion_pred'] = pose_motion_pred

        pose_pred = image_exp_pose[:, :1, -6:] + pose_motion_pred  # [1, 200, 6]

        batch['pose_pred'] = pose_pred # size() -- [1, 200, 6] 
        
        # debug_var("Audio2Pose.batch", batch)
        # Audio2Pose.batch is dict:
        # tensor pose size: [1, 6] , min: tensor(-0.0195, device='cuda:0') , max: tensor(0.2540, device='cuda:0')
        # tensor class size: [1] , min: tensor(0, device='cuda:0') , max: tensor(0, device='cuda:0')
        # tensor z size: [1, 64] , min: tensor(-1.8347, device='cuda:0') , max: tensor(1.8609, device='cuda:0')
        # tensor audio_emb size: [1, 32, 512] , min: tensor(0., device='cuda:0') , max: tensor(10.3144, device='cuda:0')
        # tensor pose_pred size: [1, 200, 6] , min: tensor(-0.0595, device='cuda:0') , max: tensor(0.5657, device='cuda:0')

        return batch
