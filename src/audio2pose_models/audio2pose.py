import torch
from torch import nn
from src.audio2pose_models.cvae import CVAE
from src.audio2pose_models.discriminator import PoseSequenceDiscriminator
from src.audio2pose_models.audio_encoder import AudioEncoder
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
        batch = {}
        ref = x['ref'] # size() -- [1, 200, 70]
        batch['ref'] = x['ref'][:, 0, -6:]  # [1, 200, 70] ==> [1, 6] , audio pose ???
        batch['class'] = x['class'] # tensor([0], device='cuda:0')
        bs = ref.shape[0]
        
        indiv_mels= x['indiv_mels'] # size() -- [1, 200, 1, 80, 16]
        indiv_mels_use = indiv_mels[:, 1:] # size() -- [1, 199, 1, 80, 16], we regard the ref as the first frame
        num_frames = x['num_frames'] # 200
        num_frames = int(num_frames) - 1 # ==> 199

        div = num_frames//self.seq_len
        re = num_frames%self.seq_len
        audio_emb_list = []
        pose_motion_pred_list = [torch.zeros(batch['ref'].unsqueeze(1).shape, dtype=batch['ref'].dtype, 
                                                device=batch['ref'].device)]

        for i in range(div):
            z = torch.randn(bs, self.latent_dim).to(ref.device)
            batch['z'] = z
            audio_emb = self.audio_encoder(indiv_mels_use[:, i*self.seq_len:(i+1)*self.seq_len,:,:,:]) #bs seq_len 512
            batch['audio_emb'] = audio_emb

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            batch = self.netG(batch)
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            pose_motion_pred_list.append(batch['pose_motion_pred'])  #list of bs seq_len 6
        
        if re != 0:
            z = torch.randn(bs, self.latent_dim).to(ref.device)
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

        pose_pred = ref[:, :1, -6:] + pose_motion_pred  # bs T 6

        batch['pose_pred'] = pose_pred # size() -- [1, 200, 6] 
        
        # (Pdb) for k, v in batch.items(): print('  ' + k + '.size():', list(v.size()))
        #   ref.size(): [1, 6]
        #   class.size(): [1]
        #   z.size(): [1, 64]
        #   audio_emb.size(): [1, 32, 512]
        #   pose_motion_pred.size(): [1, 200, 6]
        #   pose_pred.size(): [1, 200, 6]

        return batch
