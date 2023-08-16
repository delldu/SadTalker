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
import os 
import torch
import torch.nn as nn

from scipy.io import savemat
from scipy.signal import savgol_filter # xxxx8888

from SAD.audio2pose import Audio2Pose
from SAD.audio2exp import Audio2Exp
from SAD.debug import debug_var

import pdb


class Audio2Coeff(nn.Module):
    def __init__(self):
        super(Audio2Coeff, self).__init__()
        self.audio2pose_model = Audio2Pose()
        self.audio2exp_model = Audio2Exp()

    # xxxx9999 Step 2
    def generate(self, batch, coeff_save_dir, pose_style):
        # debug_var("Audio2Coeff.batch", batch)
        # Audio2Coeff.batch is dict:
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

            # exp_pred.size() -- [1, 200, 64]
            # pose_pred.size() -- [1, 200, 6]
            coeffs_pred = torch.cat((exp_pred, pose_pred), dim=-1)
            # ==> coeffs_pred.size() -- [1, 200, 70]

            coeffs_pred_numpy = coeffs_pred[0].clone().detach().cpu().numpy() 
            # coeffs_pred_numpy.shape -- (200, 70)

            savemat(os.path.join(coeff_save_dir, '%s##%s.mat'%(batch['image_name'], batch['audio_name'])),  
                    {'coeff_3dmm': coeffs_pred_numpy})

            return os.path.join(coeff_save_dir, '%s##%s.mat'%(batch['image_name'], batch['audio_name']))
    

if __name__ == "__main__":
    model = Audio2Coeff()
    print(model)