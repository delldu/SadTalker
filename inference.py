from glob import glob
import shutil
import torch
from time import  strftime
import os, sys, time
from argparse import ArgumentParser

from src.utils.preprocess import CropAndExtract
from src.audio2coeff import AudioCoeffModel  
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path
from src.utils.debug import debug_var
import pdb

def main(args):
    pic_path = args.source_image
    audio_path = args.driven_audio
    save_dir = os.path.join(args.result_dir, strftime("%Y_%m_%d_%H.%M.%S"))
    os.makedirs(save_dir, exist_ok=True)
    pose_style = args.pose_style
    device = args.device
    batch_size = args.batch_size

    current_root_path = os.path.split(sys.argv[0])[0]

    sadtalker_paths = init_path(args.checkpoint_dir, os.path.join(current_root_path, 'src/config'),
                                args.size, args.preprocess)

    #init model
    preprocess_model = CropAndExtract(sadtalker_paths, device)

    audio_to_coeff = AudioCoeffModel(sadtalker_paths,  device)
    
    animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device)

    #crop image and extract 3dmm from image
    first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)
    print('3DMM Extraction for source image')
    image_coeff_path, crop_pic_path, crop_info = preprocess_model.generate_from_image(
            pic_path, first_frame_dir, args.preprocess, 
            source_image_flag=True, pic_size=args.size)

    if image_coeff_path is None:
        print("Can't get the coeffs of the input")
        return

    #audio2ceoff
    batch = get_data(image_coeff_path, audio_path, device)
    # debug_var("audio2ceoff.batch", batch)
    # audio2ceoff.batch is dict:
    #     tensor audio_mels size: [1, 200, 1, 80, 16] , min: tensor(-4., device='cuda:0') , max: tensor(2.5998, device='cuda:0')
    #     tensor image_exp_pose size: [1, 200, 70] , min: tensor(-1.0968, device='cuda:0') , max: tensor(1.1307, device='cuda:0')
    #     audio_num_frames value: 200
    #     tensor audio_ratio size: [1, 200, 1] , min: tensor(0., device='cuda:0') , max: tensor(1., device='cuda:0')
    #     audio_name value: 'chinese_news'
    #     image_name value: 'dell'

    coeff_path = audio_to_coeff.generate(batch, save_dir, pose_style)

    #coeff2video
    data = get_facerender_data(coeff_path, crop_pic_path, image_coeff_path, audio_path, batch_size,
                expression_scale=args.expression_scale, preprocess=args.preprocess, size=args.size)
    # debug_var("coeff2video.data", data)
    # coeff2video.data is dict:
    #     tensor source_image size: [2, 3, 256, 256] , min: tensor(0.1216) , max: tensor(1.)
    #     tensor image_semantics size: [2, 70, 27] , min: tensor(-1.0968) , max: tensor(1.1307)
    #     audio_frame_num value: 200
    #     tensor audio_semantics size: [2, 100, 70, 27] , min: tensor(-1.5285) , max: tensor(1.0894)
    #     video_name value: 'dell##chinese_news'
    #     audio_path value: 'examples/driven_audio/chinese_news.wav'
    
    result = animate_from_coeff.generate(data, save_dir, pic_path, crop_info, 
                preprocess=args.preprocess, img_size=args.size)
    
    shutil.move(result, save_dir+'.mp4')
    print('The generated video is named:', save_dir+'.mp4')

    if not args.verbose:
        shutil.rmtree(save_dir)

    
if __name__ == '__main__':

    parser = ArgumentParser()  
    parser.add_argument("--driven_audio", default='./examples/driven_audio/bus_chinese.wav', help="path to driven audio")
    parser.add_argument("--source_image", default='./examples/source_image/full_body_1.png', help="path to source image")
    parser.add_argument("--checkpoint_dir", default='./checkpoints', help="path to output")
    parser.add_argument("--result_dir", default='./results', help="path to output")
    parser.add_argument("--pose_style", type=int, default=0,  help="input pose style from [0, 46)")
    parser.add_argument("--batch_size", type=int, default=2,  help="the batch size of facerender")
    parser.add_argument("--size", type=int, default=512,  help="the image size of the facerender")
    parser.add_argument("--expression_scale", type=float, default=1.,  help="the batch size of facerender")
    parser.add_argument("--cpu", dest="cpu", action="store_true") 
    parser.add_argument("--preprocess", default='crop', choices=['crop', 'extcrop', 'resize', 'full', 'extfull'], help="how to preprocess the images" ) 
    parser.add_argument("--verbose",action="store_true", help="saving the intermedia output or not" ) 

    # net structure and parameters
    parser.add_argument('--net_recon', type=str, default='resnet50', choices=['resnet50'], help='useless')
    parser.add_argument('--use_last_fc',default=False, help='zero initialize the last fc')
    parser.add_argument('--bfm_folder', type=str, default='./checkpoints/BFM_Fitting/')
    parser.add_argument('--bfm_model', type=str, default='BFM_model_front.mat', help='bfm model')

    # default renderer parameters
    # parser.add_argument('--focal', type=float, default=1015.)
    # parser.add_argument('--center', type=float, default=112.)
    # parser.add_argument('--camera_d', type=float, default=10.)
    # parser.add_argument('--z_near', type=float, default=5.)
    # parser.add_argument('--z_far', type=float, default=15.)

    args = parser.parse_args()

    if torch.cuda.is_available() and not args.cpu:
        args.device = "cuda"
    else:
        args.device = "cpu"

    main(args)

