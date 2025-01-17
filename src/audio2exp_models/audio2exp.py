from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from src.utils.debug import debug_var
from typing import Dict
import pdb

class Audio2Exp(nn.Module):
    def __init__(self, device):
        super(Audio2Exp, self).__init__()
        self.netG = Audio2ExpWrapperV2().to(device)
        # torch.jit.script(self) ==> batch ???
        # torch.jit.script(self.netG) ==> OK

    def forward(self, batch: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        # debug_var("Audio2Exp.batch", batch)
        # Audio2Exp.batch is dict:
        #     tensor audio_mels size: [1, 200, 1, 80, 16] , min: tensor(-4., device='cuda:0') , max: tensor(2.5998, device='cuda:0')
        #     tensor image_exp_pose size: [1, 200, 70] , min: tensor(-1.0968, device='cuda:0') , max: tensor(1.1307, device='cuda:0')
        ##    num_frames value: 200
        #     tensor audio_ratio size: [1, 200, 1] , min: tensor(0., device='cuda:0') , max: tensor(1., device='cuda:0')
        #     audio_name value: 'chinese_news'
        #     image_name value: 'dell'

        mel_input = batch['audio_mels'] # [1, 200, 1, 80, 16]
        bs = mel_input.shape[0]
        T = mel_input.shape[1] # [200, 1, 80, 16], T -- batch size

        exp_coeff_pred = []

        for i in tqdm(range(0, T, 10),'audio2exp:'): # every 10 frames
            current_mel_input = mel_input[:,i:i+10]
            audio_mel = current_mel_input.view(-1, 1, 80, 16) # size() -- [10, 1, 80, 16]

            image_exp = batch['image_exp_pose'][:, :, :64][:, i:i+10] # size() -- [1, 10, 64]
            audio_ratio = batch['audio_ratio'][:, i:i+10]  # size() -- [1, 10, 1]

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # self.netG -- Audio2ExpWrapperV2(...)
            curr_exp_coeff_pred  = self.netG(audio_mel, image_exp, audio_ratio) # [1, 200, 64]
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!            

            exp_coeff_pred += [curr_exp_coeff_pred]

        # BS x T x 64
        results_dict = {
            'exp_coeff_pred': torch.cat(exp_coeff_pred, axis=1) # size() -- [1, 200, 64]
            }
        
        return results_dict


class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, use_act = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual
        self.use_act = use_act

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        
        if self.use_act:
            return self.act(out)
        else:
            return out

class Audio2ExpWrapperV2(nn.Module):
    def __init__(self) -> None:
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
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            )

        self.mapping1 = nn.Linear(512+64+1, 64)
        nn.init.constant_(self.mapping1.bias, 0.)

    def forward(self, audio_mel, image_exp, audio_ratio):
        # audio_mel.size() -- [10, 1, 80, 16]
        audio_mel = self.audio_encoder(audio_mel).view(audio_mel.size(0), -1) # size() -- [10, 512]
        ref_reshape = image_exp.reshape(audio_mel.size(0), -1) # size() -- [10, 64]
        audio_ratio = audio_ratio.reshape(audio_mel.size(0), -1) # size() -- [10, 1]
        
        y = self.mapping1(torch.cat([audio_mel, ref_reshape, audio_ratio], dim=1)) # size() -- [10, 64]
        out = y.reshape(image_exp.shape[0], image_exp.shape[1], -1) # size() -- [1, 10, 64]

        return out


