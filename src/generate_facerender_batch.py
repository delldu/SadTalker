import os
import numpy as np
from PIL import Image
from skimage import io, img_as_float32, transform # xxxx8888
import torch
import scipy.io as scio
from src.utils.debug import debug_var
import pdb

def get_facerender_data(coeff_path, pic_path, image_coeff_path, audio_path, batch_size,
                        expression_scale=1.0, preprocess='crop', size = 256):
    # coeff_path = './results/2023_08_13_10.45.38/dell##chinese_news.mat'
    # pic_path = './results/2023_08_13_10.45.38/first_frame_dir/dell.png'
    # image_coeff_path = './results/2023_08_13_10.45.38/first_frame_dir/dell.mat'
    # audio_path = 'examples/driven_audio/chinese_news.wav'
    # batch_size = 2

    semantic_radius = 13 # 2-13 is lower power and important, others is almost noise !!!
    video_name = os.path.splitext(os.path.split(coeff_path)[-1])[0]
    # video_name -- 'dell##chinese_news'
    txt_path = os.path.splitext(coeff_path)[0]
    # txt_path -- './results/2023_08_13_10.45.38/dell##chinese_news'

    data={}

    img1 = Image.open(pic_path)
    source_image = np.array(img1)
    source_image = img_as_float32(source_image)
    source_image = transform.resize(source_image, (size, size, 3))
    source_image = source_image.transpose((2, 0, 1))
    source_image_ts = torch.FloatTensor(source_image).unsqueeze(0)
    
    source_image_ts = source_image_ts.repeat(batch_size, 1, 1, 1)
    data['source_image'] = source_image_ts
 
    image_coeff_dict = scio.loadmat(image_coeff_path)
    # image_coeff_dict['coeff_3dmm'].shape -- (1, 73)

    audio_coeff_dict = scio.loadmat(coeff_path)
    # audio_coeff_dict['coeff_3dmm'].shape -- (200, 70)

    if 'full' not in preprocess.lower(): # True !!!
        source_semantics = image_coeff_dict['coeff_3dmm'][:1,:70] #1 70
        audio_exp_pose = audio_coeff_dict['coeff_3dmm'][:,:70]
    else: # full mode !!!
        source_semantics = image_coeff_dict['coeff_3dmm'][:1,:73]
        audio_exp_pose = audio_coeff_dict['coeff_3dmm'][:,:70]

    source_semantics_new = transform_semantic(source_semantics, semantic_radius)
    source_semantics_ts = torch.FloatTensor(source_semantics_new).unsqueeze(0)
    source_semantics_ts = source_semantics_ts.repeat(batch_size, 1, 1)
    data['source_semantics'] = source_semantics_ts

    # target 
    audio_exp_pose[:, :64] = audio_exp_pose[:, :64] * expression_scale # expression_scale -- 1.0

    if 'full' in preprocess.lower():
        audio_exp_pose = np.concatenate([audio_exp_pose, np.repeat(source_semantics[:,70:], audio_exp_pose.shape[0], axis=0)], axis=1)

    with open(txt_path+'.txt', 'w') as f:
        for coeff in audio_exp_pose:
            for i in coeff:
                f.write(str(i)[:7]   + '  '+'\t')
            f.write('\n')

    target_semantics_list = [] 
    audio_frame_num = audio_exp_pose.shape[0] # 200
    data['audio_frame_num'] = audio_frame_num
    for frame_idx in range(audio_frame_num):
        target_semantics = transform_semantic_target(audio_exp_pose, frame_idx, semantic_radius)
        target_semantics_list.append(target_semantics)

    remainder = audio_frame_num%batch_size
    if remainder != 0:
        for _ in range(batch_size-remainder):
            target_semantics_list.append(target_semantics)

    target_semantics_np = np.array(target_semantics_list)
    target_semantics_np = target_semantics_np.reshape(batch_size, -1, 
                                target_semantics_np.shape[-2], target_semantics_np.shape[-1])
    data['target_semantics'] = torch.FloatTensor(target_semantics_np)
    data['video_name'] = video_name
    data['audio_path'] = audio_path

    # debug_var("get_facerender_data.data", data)
    # get_facerender_data.data is dict:
    #     tensor source_image size: [2, 3, 256, 256] , min: tensor(0.1216) , max: tensor(1.)
    #     tensor source_semantics size: [2, 70, 27] , min: tensor(-1.0968) , max: tensor(1.1307)
    #     audio_frame_num value: 200
    #     tensor target_semantics size: [2, 100, 70, 27] , min: tensor(-1.6630) , max: tensor(1.0894)
    #     video_name value: 'dell##chinese_news'
    #     audio_path value: 'examples/driven_audio/chinese_news.wav'

    return data

def transform_semantic(semantic, semantic_radius: int):
    # array semantic shape: (1, 70) , min: -1.0967898 , max: 1.13074
    semantic_list =  [semantic for i in range(0, semantic_radius*2+1)]
    coeff_3dmm = np.concatenate(semantic_list, 0) # shape: (27, 70)
    return coeff_3dmm.transpose(1, 0) # ==> shape: (70, 27)

def transform_semantic_target(coeff_3dmm, frame_index: int, semantic_radius: int):
    # array coeff_3dmm shape: (200, 70) , min: -1.6095467 , max: 1.0893884

    audio_num_frames = coeff_3dmm.shape[0]
    seq = list(range(frame_index- semantic_radius, frame_index + semantic_radius+1))
    # seq -- [-13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 
    #     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    index = [ min(max(item, 0), audio_num_frames-1) for item in seq ] 
    # index -- [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    coeff_3dmm_g = coeff_3dmm[index, :] # shape -- (27, 70)
    return coeff_3dmm_g.transpose(1,0) # ==> shape -- (70, 27)

