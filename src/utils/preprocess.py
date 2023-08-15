import numpy as np
import cv2, os, torch
from tqdm import tqdm
from PIL import Image 

import safetensors.torch 
from src.face3d.util.preprocess import align_img
from src.face3d.util.load_mats import load_lm3d
from src.face3d.models import networks

from scipy.io import savemat
from src.utils.croper import Preprocesser
from src.utils.safetensor_helper import load_x_from_safetensor 
import pdb

def split_coeff(coeffs):
        """
        Return:
            coeffs_dict     -- a dict of torch.tensors

        Parameters:
            coeffs          -- torch.tensor, size (B, 256)
        """
        id_coeffs = coeffs[:, :80]
        exp_coeffs = coeffs[:, 80: 144]
        tex_coeffs = coeffs[:, 144: 224]
        angles = coeffs[:, 224: 227]
        gammas = coeffs[:, 227: 254]
        translations = coeffs[:, 254:]
        return {
            'id': id_coeffs,
            'exp': exp_coeffs,
            'tex': tex_coeffs,
            'angle': angles,
            'gamma': gammas,
            'trans': translations
        }


class CropAndExtract(): # xxxx8888
    def __init__(self, sadtalker_path, device):

        self.propress = Preprocesser(device)
        self.net_recon = networks.define_net_recon(net_recon='resnet50', use_last_fc=False, init_path='').to(device)
        
        # sadtalker_path['checkpoint'] -- './checkpoints/SadTalker_V0.0.2_256.safetensors'
        checkpoint = safetensors.torch.load_file(sadtalker_path['checkpoint'])    
        self.net_recon.load_state_dict(load_x_from_safetensor(checkpoint, 'face_3drecon'))

        self.net_recon.eval()
        self.lm3d_std = load_lm3d(sadtalker_path['dir_of_BFM_fitting']) # shape (5, 3)
        # sadtalker_path['dir_of_BFM_fitting'] -- 'src/config'

        self.device = device
    
    def generate_from_image(self, input_path, save_dir, crop_or_resize='crop', source_image_flag=False, pic_size=256):
        image_name = os.path.splitext(os.path.split(input_path)[-1])[0]  

        landmarks_path =  os.path.join(save_dir, image_name+'_landmarks.txt') 
        image_coeff_path =  os.path.join(save_dir, image_name+'.mat')  
        png_path =  os.path.join(save_dir, image_name+'.png')  

        #load input
        if not os.path.isfile(input_path):
            raise ValueError('input_path must be a valid path to video/image file')
        elif input_path.split('.')[-1] in ['jpg', 'png', 'jpeg']:
            # loader for first frame
            full_frames = [cv2.imread(input_path)]
            fps = 25
        else:
            # loader for videos
            video_stream = cv2.VideoCapture(input_path)
            fps = video_stream.get(cv2.CAP_PROP_FPS)
            full_frames = [] 
            while 1:
                still_reading, frame = video_stream.read()
                if not still_reading:
                    video_stream.release()
                    break 
                full_frames.append(frame) 
                if source_image_flag:
                    break

        x_full_frames= [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  for frame in full_frames] 

        #### crop images as the 
        if 'crop' in crop_or_resize.lower(): # default crop
            x_full_frames, crop, quad = self.propress.crop(x_full_frames, still=True if 'ext' in crop_or_resize.lower() else False, xsize=512)
            clx, cly, crx, cry = crop
            lx, ly, rx, ry = quad
            lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
            oy1, oy2, ox1, ox2 = cly+ly, cly+ry, clx+lx, clx+rx
            crop_info = ((ox2 - ox1, oy2 - oy1), crop, quad)
        elif 'full' in crop_or_resize.lower():
            x_full_frames, crop, quad = self.propress.crop(x_full_frames, still=True if 'ext' in crop_or_resize.lower() else False, xsize=512)
            clx, cly, crx, cry = crop
            lx, ly, rx, ry = quad
            lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
            oy1, oy2, ox1, ox2 = cly+ly, cly+ry, clx+lx, clx+rx
            crop_info = ((ox2 - ox1, oy2 - oy1), crop, quad)
        else: # resize mode
            oy1, oy2, ox1, ox2 = 0, x_full_frames[0].shape[0], 0, x_full_frames[0].shape[1] 
            crop_info = ((ox2 - ox1, oy2 - oy1), None, None)

        frames_pil = [Image.fromarray(cv2.resize(frame,(pic_size, pic_size))) for frame in x_full_frames]
        if len(frames_pil) == 0:
            print('No face is detected in the input file')
            return None, None

        # save crop info
        for frame in frames_pil:
            cv2.imwrite(png_path, cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))

        # 2. get the landmark according to the detected face. 
        if not os.path.isfile(landmarks_path): 
            lm = self.propress.predictor.extract_keypoint(frames_pil, landmarks_path)
        else:
            print(' Using saved landmarks.')
            lm = np.loadtxt(landmarks_path).astype(np.float32)
            lm = lm.reshape([len(x_full_frames), -1, 2])

        if not os.path.isfile(image_coeff_path):
            # load 3dmm paramter generator from Deep3DFaceRecon_pytorch 
            video_coeffs, full_coeffs = [],  []
            for idx in tqdm(range(len(frames_pil)), desc='3DMM Extraction In Video:'):
                frame = frames_pil[idx] # PIL.Image.Image image mode=RGB size=256x256
                W,H = frame.size
                lm1 = lm[idx].reshape([-1, 2])
            
                if np.mean(lm1) == -1: # NO face !!!
                    lm1 = (self.lm3d_std[:, :2]+1)/2. # (x,y,z) ==> (x, y)
                    lm1 = np.concatenate([lm1[:, :1]*W, lm1[:, 1:2]*H], 1)
                else:
                    lm1[:, -1] = H - 1 - lm1[:, -1]

                trans_params, im1, lm1, _ = align_img(frame, lm1, self.lm3d_std)
 
                trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)]).astype(np.float32)
                im_t = torch.tensor(np.array(im1)/255., dtype=torch.float32).permute(2, 0, 1).to(self.device).unsqueeze(0)

                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # im1 -- <PIL.Image.Image image mode=RGB size=224x224>
                # im_t.size() -- [1, 3, 224, 224]

                # xxxx1111 image coeff model
                with torch.no_grad():
                    full_coeff = self.net_recon(im_t) # ImageCoeffModel(...)
                    coeffs = split_coeff(full_coeff)

                pred_coeff = {key:coeffs[key].cpu().numpy() for key in coeffs}
                # full_coeff.size() -- [1, 257]
                # (Pdb) for k, v in coeffs.items(): print(k, ":", list(v.size()))
                # id : [1, 80]
                # exp : [1, 64]
                # tex : [1, 80]
                # angle : [1, 3]
                # gamma : [1, 27]
                # trans : [1, 3]
                # trans_params -- array([256., 256., 0.99591, 129.40619, 105.512375], dtype=float32)
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!                
        
                pred_coeff = np.concatenate([
                    pred_coeff['exp'], 
                    pred_coeff['angle'],
                    pred_coeff['trans'],
                    trans_params[2:][None], # array([[  0.99591 , 129.40619 , 105.512375]], dtype=float32) ???
                    ], 1)
                video_coeffs.append(pred_coeff)
                full_coeffs.append(full_coeff.cpu().numpy()) # useless

            semantic_npy = np.array(video_coeffs)[:, 0] 
            image_coeff_dict = {'coeff_3dmm': semantic_npy, 'full_3dmm': np.array(full_coeffs)[0]}
            savemat(image_coeff_path, image_coeff_dict)

        return image_coeff_path, png_path, crop_info
