from tqdm import tqdm
import torch
from torch import nn
import pdb

class Audio2Exp(nn.Module):
    def __init__(self, netG, device):
        super(Audio2Exp, self).__init__()
        self.device = device
        self.netG = netG.to(device)

    def forward(self, batch):
        # batch --
        # {   'indiv_mels': indiv_mels,  # len() -- 1, indiv_mels[0].size() -- [200, 1, 80, 16]
        #     'ref': ref_coeff,  # size() -- [1, 200, 70]
        #     'num_frames': num_frames, # -- 200
        #     'ratio_gt': ratio, # [1, 200, 1], 
        #     'audio_name': audio_name, # 'chinese_news'
        #     'pic_name': pic_name, # 'dell'
        # }

        mel_input = batch['indiv_mels']                         # bs T 1 80 16
        bs = mel_input.shape[0]
        T = mel_input.shape[1] # [200, 1, 80, 16], T -- batch size

        exp_coeff_pred = []

        for i in tqdm(range(0, T, 10),'audio2exp:'): # every 10 frames
            current_mel_input = mel_input[:,i:i+10]
            audiox = current_mel_input.view(-1, 1, 80, 16) # bs*T 1 80 16

            ref = batch['ref'][:, :, :64][:, i:i+10] # ref: exp -- 64
            ratio = batch['ratio_gt'][:, i:i+10]  #bs T, xxxx8888 ratio_gt seems random ???

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!            
            curr_exp_coeff_pred  = self.netG(audiox, ref, ratio) # bs T 64 ==> [1, 200, 64]
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!            

            exp_coeff_pred += [curr_exp_coeff_pred]

        # BS x T x 64
        results_dict = {
            'exp_coeff_pred': torch.cat(exp_coeff_pred, axis=1)
            }
        return results_dict


