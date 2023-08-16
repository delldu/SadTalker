import os 
import torch
import numpy as np
from scipy.io import savemat
from scipy.signal import savgol_filter # xxxx8888

import safetensors
import safetensors.torch 

from src.audio2pose_models.audio2pose import Audio2Pose
from src.audio2exp_models.audio2exp import Audio2Exp
from src.utils.safetensor_helper import load_x_from_safetensor
from src.utils.debug import debug_var

import pdb

# xxxx8888 *************************************************************************
class AudioCoeffModel():
    def __init__(self, sadtalker_path, device):
        # load audio2pose_model
        self.audio2pose_model = Audio2Pose(device=device)
        self.audio2pose_model = self.audio2pose_model.to(device)
        self.audio2pose_model.eval()
        for param in self.audio2pose_model.parameters():
            param.requires_grad = False 
        
        try:
            # './checkpoints/SadTalker_V0.0.2_256.safetensors'
            checkpoints = safetensors.torch.load_file(sadtalker_path['checkpoint'])
            self.audio2pose_model.load_state_dict(load_x_from_safetensor(checkpoints, 'audio2pose', 
                skip_key1='netG.encoder.', skip_key2='netD_motion.seq.'))
            # torch.save(self.audio2pose_model.state_dict(), "/tmp/Audio2Pose.pth") # xxxx3333
        except:
            raise Exception("Failed in loading audio2pose_checkpoint")

        self.audio2exp_model = Audio2Exp(device=device)
        self.audio2exp_model = self.audio2exp_model.to(device)
        for param in self.audio2exp_model.parameters():
            param.requires_grad = False
        self.audio2exp_model.eval()
        try:
            checkpoints = safetensors.torch.load_file(sadtalker_path['checkpoint'])
            self.audio2exp_model.netG.load_state_dict(load_x_from_safetensor(checkpoints, 'audio2exp'))
            # torch.save(self.audio2exp_model.state_dict(), "/tmp/Audio2Exp.pth") # xxxx3333
        except:
            raise Exception("Failed in loading audio2exp_checkpoint")

        self.device = device

    # xxxx9999 Step 2
    def generate(self, batch, coeff_save_dir, pose_style):
        # debug_var("AudioCoeffModel.batch", batch)
        # AudioCoeffModel.batch is dict:
        #     tensor audio_mels size: [1, 200, 1, 80, 16] , min: tensor(-4., device='cuda:0') , max: tensor(2.5998, device='cuda:0')
        #     tensor image_exp_pose size: [1, 200, 70] , min: tensor(-1.0968, device='cuda:0') , max: tensor(1.1307, device='cuda:0')
        #     audio_num_frames value: 200
        #     tensor audio_ratio size: [1, 200, 1] , min: tensor(0., device='cuda:0') , max: tensor(1., device='cuda:0')
        #     audio_name value: 'chinese_news'
        #     image_name value: 'dell'

        with torch.no_grad():
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            results_dict_exp= self.audio2exp_model(batch) # Audio2Exp(...)
            exp_pred = results_dict_exp['exp_coeff_pred'] # exp_pred.size() -- [1, 200, 64]
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            #for class_id in  range(1):
            #class_id = 0#(i+10)%45
            #class_id = random.randint(0,46)                                   #46 styles can be selected 
            batch['class'] = torch.LongTensor([pose_style]).to(self.device)

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            results_dict_pose = self.audio2pose_model(batch) # Audio2Pose(...)
            pose_pred = results_dict_pose['pose_pred'] # size() -- [1, 200, 6]
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            # xxxx8888
            pose_len = pose_pred.shape[1]
            if pose_len<13: # False
                pose_len = int((pose_len-1)/2)*2+1
                pose_pred = torch.Tensor(savgol_filter(np.array(pose_pred.cpu()), pose_len, 2, axis=1)).to(self.device)
            else:
                pose_pred = torch.Tensor(savgol_filter(np.array(pose_pred.cpu()), 13, 2, axis=1)).to(self.device) 

            # exp_pred.size() -- [1, 200, 64]
            # pose_pred.size() -- [1, 200, 6]
            coeffs_pred = torch.cat((exp_pred, pose_pred), dim=-1)
            # ==> coeffs_pred.size() -- [1, 200, 70]

            coeffs_pred_numpy = coeffs_pred[0].clone().detach().cpu().numpy() 
            # coeffs_pred_numpy.shape -- (200, 70)

            savemat(os.path.join(coeff_save_dir, '%s##%s.mat'%(batch['image_name'], batch['audio_name'])),  
                    {'coeff_3dmm': coeffs_pred_numpy})

            return os.path.join(coeff_save_dir, '%s##%s.mat'%(batch['image_name'], batch['audio_name']))
    
