import os
import cv2
import yaml
import numpy as np
import warnings
from skimage import img_as_ubyte
import safetensors
import safetensors.torch 
warnings.filterwarnings('ignore')


import imageio
import torch
import torchvision


# from src.facerender.modules.keypoint_detector import HEEstimator, KPDetector
from src.facerender.modules.keypoint_detector import KPDetector
from src.facerender.modules.mapping import MappingNet
# from src.facerender.modules.generator import OcclusionAwareGenerator, OcclusionAwareSPADEGenerator
from src.facerender.modules.generator import OcclusionAwareSPADEGenerator
from src.facerender.modules.make_animation import make_animation 

from pydub import AudioSegment 
from src.utils.face_enhancer import enhancer_generator_with_len, enhancer_list
from src.utils.paste_pic import paste_pic
from src.utils.videoio import save_video_with_watermark

import pdb
try:
    import webui  # in webui
    in_webui = True
except:
    in_webui = False

class AnimateFromCoeff():

    def __init__(self, sadtalker_path, device):

        with open(sadtalker_path['facerender_yaml']) as f:
            config = yaml.safe_load(f)

        generator = OcclusionAwareSPADEGenerator(**config['model_params']['generator_params'],
                                                    **config['model_params']['common_params'])
        kp_extractor = KPDetector(**config['model_params']['kp_detector_params'],
                                    **config['model_params']['common_params'])
        mapping = MappingNet(**config['model_params']['mapping_params'])

        generator.to(device)
        kp_extractor.to(device)
        mapping.to(device)
        for param in generator.parameters():
            param.requires_grad = False
        for param in kp_extractor.parameters():
            param.requires_grad = False 
        for param in mapping.parameters():
            param.requires_grad = False

        if sadtalker_path is not None:
            if 'checkpoint' in sadtalker_path: # use safe tensor
                self.load_cpk_facevid2vid_safetensor(sadtalker_path['checkpoint'], kp_detector=kp_extractor, 
                    generator=generator)
            else:
                self.load_cpk_facevid2vid(sadtalker_path['free_view_checkpoint'], kp_detector=kp_extractor, 
                    generator=generator)
        else:
            raise AttributeError("Checkpoint should be specified for video head pose estimator.")

        # sadtalker_path['mappingnet_checkpoint'] -- './checkpoints/mapping_00229-model.pth.tar'
        if  sadtalker_path['mappingnet_checkpoint'] is not None:
            self.load_cpk_mapping(sadtalker_path['mappingnet_checkpoint'], mapping=mapping)
        else:
            raise AttributeError("Checkpoint should be specified for video head pose estimator.") 

        self.kp_extractor = kp_extractor
        self.generator = generator
        self.mapping = mapping

        self.kp_extractor.eval()
        self.generator.eval()
        self.mapping.eval()
         
        self.device = device
    

    def load_cpk_facevid2vid_safetensor(self, checkpoint_path, generator=None, 
                        kp_detector=None, device="cpu"):
        # checkpoint_path = './checkpoints/SadTalker_V0.0.2_256.safetensors'
        # generator = OcclusionAwareSPADEGenerator(...)
        # kp_detector = KPDetector(...)

        checkpoint = safetensors.torch.load_file(checkpoint_path)

        if generator is not None: # True
            x_generator = {}
            for k,v in checkpoint.items():
                if 'generator' in k:
                    x_generator[k.replace('generator.', '')] = v
            generator.load_state_dict(x_generator)
        if kp_detector is not None: # True
            x_generator = {}
            for k,v in checkpoint.items():
                if 'kp_extractor' in k:
                    x_generator[k.replace('kp_extractor.', '')] = v
            kp_detector.load_state_dict(x_generator)
        
        return None

    # def load_cpk_facevid2vid(self, checkpoint_path, generator=None, discriminator=None, 
    #                     kp_detector=None, optimizer_generator=None, 
    #                     optimizer_discriminator=None, optimizer_kp_detector=None, 
    #                     optimizer_he_estimator=None, device="cpu"):
    #     checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    #     if generator is not None:
    #         generator.load_state_dict(checkpoint['generator'])
    #     if kp_detector is not None:
    #         kp_detector.load_state_dict(checkpoint['kp_detector'])
    #     if discriminator is not None:
    #         try:
    #            discriminator.load_state_dict(checkpoint['discriminator'])
    #         except:
    #            print ('No discriminator in the state-dict. Dicriminator will be randomly initialized')
    #     if optimizer_generator is not None:
    #         optimizer_generator.load_state_dict(checkpoint['optimizer_generator'])
    #     if optimizer_discriminator is not None:
    #         try:
    #             optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator'])
    #         except RuntimeError as e:
    #             print ('No discriminator optimizer in the state-dict. Optimizer will be not initialized')
    #     if optimizer_kp_detector is not None:
    #         optimizer_kp_detector.load_state_dict(checkpoint['optimizer_kp_detector'])
    #     if optimizer_he_estimator is not None:
    #         optimizer_he_estimator.load_state_dict(checkpoint['optimizer_he_estimator'])

    #     pdb.set_trace()

    #     return checkpoint['epoch']
    
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

        source_image=x['source_image'].type(torch.FloatTensor)
        source_semantics=x['source_semantics'].type(torch.FloatTensor)
        target_semantics=x['target_semantics_list'].type(torch.FloatTensor)
        source_image=source_image.to(self.device) # size() -- [2, 3, 256, 256]
        source_semantics=source_semantics.to(self.device)  # size() -- [2, 70, 27]
        target_semantics=target_semantics.to(self.device)  # size() -- [2, 100, 70, 27]

        frame_num = x['frame_num'] # 200

        predictions_video = make_animation(source_image, source_semantics, target_semantics,
                                        self.generator, self.kp_extractor, self.mapping, 
                                        )
        # predictions_video.size() -- [2, 100, 3, 256, 256]

        predictions_video = predictions_video.reshape((-1,)+predictions_video.shape[2:])
        predictions_video = predictions_video[:frame_num]

        video = []
        for idx in range(predictions_video.shape[0]):
            image = predictions_video[idx]
            image = np.transpose(image.data.cpu().numpy(), [1, 2, 0]).astype(np.float32)
            video.append(image)
        result = img_as_ubyte(video)

        ### the generated video is 256x256, so we keep the aspect ratio, 
        original_size = crop_info[0] # ==> (351, 350)
        if original_size:
            result = [ cv2.resize(result_i,(img_size, int(img_size * original_size[1]/original_size[0]) )) for result_i in result ]

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
        frames = frame_num 
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

