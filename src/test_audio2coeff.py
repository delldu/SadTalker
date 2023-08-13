import os 
import torch
import numpy as np
from scipy.io import savemat, loadmat
from yacs.config import CfgNode as CN
from scipy.signal import savgol_filter

import safetensors
import safetensors.torch 

from src.audio2pose_models.audio2pose import Audio2Pose
from src.audio2exp_models.networks import SimpleWrapperV2 
from src.audio2exp_models.audio2exp import Audio2Exp
from src.utils.safetensor_helper import load_x_from_safetensor  
import pdb

def load_cpk(checkpoint_path, model=None, optimizer=None, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    if model is not None:
        model.load_state_dict(checkpoint['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    pdb.set_trace()
    
    return checkpoint['epoch']

class Audio2Coeff():
    '''
        ==> (200, 70) --70 = pose(6) + exp(64)
    '''

    def __init__(self, sadtalker_path, device):
        #load config
        # sadtalker_path['audio2pose_yaml_path'] -- 'src/config/auido2pose.yaml'
        fcfg_pose = open(sadtalker_path['audio2pose_yaml_path'])
        cfg_pose = CN.load_cfg(fcfg_pose)
        cfg_pose.freeze()

        # sadtalker_path['audio2exp_yaml_path'] -- 'src/config/auido2exp.yaml'
        fcfg_exp = open(sadtalker_path['audio2exp_yaml_path'])
        cfg_exp = CN.load_cfg(fcfg_exp)
        cfg_exp.freeze()

        # load audio2pose_model
        self.audio2pose_model = Audio2Pose(cfg_pose, None, device=device)
        self.audio2pose_model = self.audio2pose_model.to(device)
        self.audio2pose_model.eval()
        for param in self.audio2pose_model.parameters():
            param.requires_grad = False 
        
        try:
            if sadtalker_path['use_safetensor']:
                # './checkpoints/SadTalker_V0.0.2_256.safetensors'
                checkpoints = safetensors.torch.load_file(sadtalker_path['checkpoint'])
                self.audio2pose_model.load_state_dict(load_x_from_safetensor(checkpoints, 'audio2pose'))
            else:
                load_cpk(sadtalker_path['audio2pose_checkpoint'], model=self.audio2pose_model, device=device)
        except:
            raise Exception("Failed in loading audio2pose_checkpoint")

        # load audio2exp_model
        netG = SimpleWrapperV2()
        netG = netG.to(device)
        for param in netG.parameters():
            netG.requires_grad = False
        netG.eval()
        try:
            if sadtalker_path['use_safetensor']:
                checkpoints = safetensors.torch.load_file(sadtalker_path['checkpoint'])
                netG.load_state_dict(load_x_from_safetensor(checkpoints, 'audio2exp'))
            else:
                load_cpk(sadtalker_path['audio2exp_checkpoint'], model=netG, device=device)
        except:
            raise Exception("Failed in loading audio2exp_checkpoint")

        self.audio2exp_model = Audio2Exp(netG, cfg_exp, device=device, prepare_training_loss=False)
        self.audio2exp_model = self.audio2exp_model.to(device)
        for param in self.audio2exp_model.parameters():
            param.requires_grad = False
        self.audio2exp_model.eval()

        # xxxx8888
 
        self.device = device

    def generate(self, batch, coeff_save_dir, pose_style):

        with torch.no_grad():
            #test

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            results_dict_exp= self.audio2exp_model.test(batch)
            exp_pred = results_dict_exp['exp_coeff_pred']                         #bs T 64
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            #for class_id in  range(1):
            #class_id = 0#(i+10)%45
            #class_id = random.randint(0,46)                                   #46 styles can be selected 
            batch['class'] = torch.LongTensor([pose_style]).to(self.device)

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            results_dict_pose = self.audio2pose_model.test(batch) 
            pose_pred = results_dict_pose['pose_pred']                        #bs T 6
            # results_dict_pose['pose_pred'].size() -- [1, 200, 6]
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            pose_len = pose_pred.shape[1]
            if pose_len<13: # False
                pose_len = int((pose_len-1)/2)*2+1
                pose_pred = torch.Tensor(savgol_filter(np.array(pose_pred.cpu()), pose_len, 2, axis=1)).to(self.device)
            else:
                pose_pred = torch.Tensor(savgol_filter(np.array(pose_pred.cpu()), 13, 2, axis=1)).to(self.device) 
            # pose_pred.size() -- [1, 200, 6]
            # exp_pred.size() -- [1, 200, 64]

            coeffs_pred = torch.cat((exp_pred, pose_pred), dim=-1)            #bs T 70
            # ==> coeffs_pred.size() -- [1, 200, 70]

            coeffs_pred_numpy = coeffs_pred[0].clone().detach().cpu().numpy() 
            # coeffs_pred_numpy.shape -- (200, 70)

            savemat(os.path.join(coeff_save_dir, '%s##%s.mat'%(batch['pic_name'], batch['audio_name'])),  
                    {'coeff_3dmm': coeffs_pred_numpy})

            return os.path.join(coeff_save_dir, '%s##%s.mat'%(batch['pic_name'], batch['audio_name']))
    
