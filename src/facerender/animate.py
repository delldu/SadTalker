import os
import cv2
import numpy as np
from skimage import img_as_ubyte
import safetensors
import safetensors.torch 

import imageio
import torch

from src.facerender.modules.keypoint_detector import KPDetector
from src.facerender.modules.mapping import MappingNet
from src.facerender.modules.generator import OcclusionAwareSPADEGenerator
from src.facerender.modules.make_animation import make_animation 

from pydub import AudioSegment  # xxxx8888
from src.utils.paste_pic import paste_pic
from src.utils.videoio import save_video_with_watermark
from src.utils.debug import debug_var

import pdb
try:
    import webui  # in webui
    in_webui = True
except:
    in_webui = False

class AnimateFromCoeff():

    def __init__(self, sadtalker_path, device):
        self.device = device

        self.generator = OcclusionAwareSPADEGenerator().to(device)
        self.generator.eval()

        self.kp_extractor = KPDetector().to(device)
        self.kp_extractor.eval()

        self.mapping = MappingNet().to(device)
        self.mapping.eval()

        for param in self.generator.parameters():
            param.requires_grad = False
        for param in self.kp_extractor.parameters():
            param.requires_grad = False 
        for param in self.mapping.parameters():
            param.requires_grad = False

        self.load_cpk_facevid2vid_safetensor(sadtalker_path['checkpoint'],
                kp_detector=self.kp_extractor, generator=self.generator)

        # sadtalker_path['mappingnet_checkpoint'] -- './checkpoints/mapping_00229-model.pth.tar'
        self.load_cpk_mapping(sadtalker_path['mappingnet_checkpoint'], mapping=self.mapping)
    

    def load_cpk_facevid2vid_safetensor(self, checkpoint_path, generator=None, 
                        kp_detector=None, device="cpu"):
        # checkpoint_path = './checkpoints/SadTalker_V0.0.2_256.safetensors'
        # generator = OcclusionAwareSPADEGenerator(...)
        # kp_detector = KPDetector(...)

        checkpoint = safetensors.torch.load_file(checkpoint_path)

        x_generator = {}
        for k,v in checkpoint.items():
            if 'generator' in k:
                x_generator[k.replace('generator.', '')] = v
        generator.load_state_dict(x_generator)

        x_generator = {}
        for k,v in checkpoint.items():
            if 'kp_extractor' in k:
                x_generator[k.replace('kp_extractor.', '')] = v
        kp_detector.load_state_dict(x_generator)
        

    def load_cpk_mapping(self, checkpoint_path, mapping, device='cpu'):
        # checkpoint_path = './checkpoints/mapping_00229-model.pth.tar'
        # mapping = MappingNet(...)

        checkpoint = torch.load(checkpoint_path,  map_location=torch.device(device))
        mapping.load_state_dict(checkpoint['mapping'])

        return checkpoint['epoch'] # 229

    def generate(self, x, video_save_dir, pic_path, crop_info, preprocess='crop', img_size=256):
        # video_save_dir = './results/2023_08_13_10.45.38'
        # pic_path = 'examples/source_image/dell.png'
        # crop_info = ((351, 350), (0, 0, 353, 353), [1.6023768164683942, 0, 352.55072987298473, 350.24273121575817])

        # debug_var("AnimateFromCoeff.x", x)
        # AnimateFromCoeff.x is dict:
        #     tensor source_image size: [2, 3, 256, 256] , min: tensor(0.1216) , max: tensor(1.)
        #     tensor source_semantics size: [2, 70, 27] , min: tensor(-1.0968) , max: tensor(1.1307)
        #     audio_frame_num value: 200
        #     tensor target_semantics size: [2, 100, 70, 27] , min: tensor(-1.6407) , max: tensor(1.0894)
        #     video_name value: 'dell##chinese_news'
        #     audio_path value: 'examples/driven_audio/chinese_news.wav'

        source_image=x['source_image'].to(self.device) # size() -- [2, 3, 256, 256]
        source_semantics=x['source_semantics'].to(self.device)  # size() -- [2, 70, 27]
        target_semantics=x['target_semantics'].to(self.device)  # size() -- [2, 100, 70, 27]
        
        audio_frame_num = x['audio_frame_num'] # 200

        predictions_video = make_animation(source_image, source_semantics, target_semantics,
                                        self.generator, self.kp_extractor, self.mapping)
        # predictions_video.size() -- [2, 100, 3, 256, 256]

        predictions_video = predictions_video.reshape((-1,) + predictions_video.shape[2:])
        predictions_video = predictions_video[:audio_frame_num]

        video = []
        for idx in range(predictions_video.shape[0]):
            image = predictions_video[idx]
            image = np.transpose(image.data.cpu().numpy(), [1, 2, 0]).astype(np.float32)
            video.append(image)
        result = img_as_ubyte(video)

        ### the generated video is 256x256, so we keep the aspect ratio, 
        original_size = crop_info[0] # ==> (351, 350)
        if original_size:
            result = [ cv2.resize(result_i,(img_size, 
                       int(img_size * original_size[1]/original_size[0]) )) for result_i in result ]

        # predictions_video.size() -- [200, 3, 256, 256]
        
        video_name = x['video_name']  + '.mp4'
        path = os.path.join(video_save_dir, 'temp_'+video_name)
        
        imageio.mimsave(path, result,  fps=float(25))

        av_path = os.path.join(video_save_dir, video_name)
        return_path = av_path 
        
        audio_path =  x['audio_path'] # 'examples/driven_audio/chinese_news.wav'
        audio_name = os.path.splitext(os.path.split(audio_path)[-1])[0]
        new_audio_path = os.path.join(video_save_dir, audio_name+'.wav')
        # './results/2023_08_13_10.45.38/chinese_news.wav'

        start_time = 0
        # cog will not keep the .mp3 filename
        sound = AudioSegment.from_file(audio_path)
        frames = audio_frame_num 
        end_time = start_time + frames*1/25*1000
        word1 = sound.set_frame_rate(16000)
        word = word1[start_time:end_time]
        word.export(new_audio_path, format="wav")

        save_video_with_watermark(path, new_audio_path, av_path, watermark= False)
        print(f'The generated video is named {video_save_dir}/{video_name}') 

        if 'full' in preprocess.lower():
            # only add watermark to the full image.
            video_name_full = x['video_name']  + '_full.mp4'
            full_video_path = os.path.join(video_save_dir, video_name_full)
            return_path = full_video_path
            paste_pic(path, pic_path, crop_info, new_audio_path, full_video_path, extended_crop= True if 'ext' in preprocess.lower() else False)
            print(f'The generated video is named {video_save_dir}/{video_name_full}') 
        else:
            full_video_path = av_path 

            os.remove(path)
        os.remove(new_audio_path)

        return return_path

