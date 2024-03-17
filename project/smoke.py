# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2020-2024(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 12月 28日 星期一 14:29:37 CST
# ***
# ************************************************************************************/
#

import os
import torch
import SAD
import argparse
import todos
import pdb

def test_input_shape():
    import time
    import random
    from tqdm import tqdm

    print("Test input shape ...")

    model, device = SAD.get_script_model()

    N = 100
    B, C, H, W = 1, 3, 256, 256

    mean_time = 0
    progress_bar = tqdm(total=N)
    for count in range(N):
        progress_bar.update(1)

        kp1 = torch.randn(B, 50, 2)
        kp2 = torch.randn(B, 50, 2)

        x = torch.randn(B, C, H, W)

        start_time = time.time()
        with torch.no_grad():
            y = model(kp1.to(device), kp2.to(device), x.to(device))
        torch.cuda.synchronize()
        mean_time += time.time() - start_time

    mean_time /= N
    print(f"Mean spend {mean_time:0.4f} seconds")
    os.system("nvidia-smi | grep python")


def run_bench_mark():
    print("Run benchmark ...")

    model, device = SAD.get_script_model()
    N = 100
    B, C, H, W = 1, 3, 256, 256

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as p:
        for ii in range(N):
            image1 = torch.randn(B, C, H, W)
            image2 = torch.randn(B, C, H, W)
            with torch.no_grad():
                y = model(image1.to(device), image2.to(device))
            torch.cuda.synchronize()
        p.step()

    print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
    os.system("nvidia-smi | grep python")

def export_image_3d_exp_pose_onnx_model():
    import onnx
    import onnxruntime
    from onnxsim import simplify
    import onnxoptimizer

    print("Export image_3d_exp_pose onnx model ...")

    # 1. Run torch model
    model, device = SAD.get_script_model()
    model = model.image2coffe_model

    B, C, H, W = 1, 3, 512, 512
    dummy_input = torch.randn(B, C, H, W).to(device)

    with torch.no_grad():
        dummy_output = model(dummy_input)
    torch_outputs = [dummy_output.cpu()]

    # 2. Export onnx model
    input_names = [ "input" ]
    output_names = [ "output" ]
    onnx_filename = "output/image_3d_exp_pose.onnx"

    torch.onnx.export(model, 
        (dummy_input),
        onnx_filename, 
        verbose=False, 
        input_names=input_names, 
        output_names=output_names,
        opset_version=16,
    )

    # 3. Check onnx model file
    onnx_model = onnx.load(onnx_filename)
    onnx.checker.check_model(onnx_model)

    onnx_model, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx_model = onnxoptimizer.optimize(onnx_model)
    onnx.save(onnx_model, onnx_filename)
    # print(onnx.helper.printable_graph(onnx_model.graph))

    # 4. Run onnx model
    if 'cuda' in device.type:
        ort_session = onnxruntime.InferenceSession(onnx_filename, providers=['CUDAExecutionProvider'])
    else:        
        ort_session = onnxruntime.InferenceSession(onnx_filename, providers=['CPUExecutionProvider'])

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    onnx_inputs = { input_names[0]: to_numpy(dummy_input), 
                }
    onnx_outputs = ort_session.run(None, onnx_inputs)

    # 5.Compare output results
    assert len(torch_outputs) == len(onnx_outputs)
    for torch_output, onnx_output in zip(torch_outputs, onnx_outputs):
        torch.testing.assert_close(torch_output, torch.tensor(onnx_output), rtol=0.01, atol=0.01)

    todos.model.reset_device()

    print("!!!!!! Torch and ONNX Runtime output matched !!!!!!")

def export_audio_3d_exp_pose_onnx_model():
    import onnx
    import onnxruntime
    from onnxsim import simplify
    import onnxoptimizer

    print("Export audio_3d_exp_pose onnx model ...")

    # 1. Run torch model
    model, device = SAD.get_script_model()
    model = model.audio2coffe_model

    audio_mels_input = torch.randn(1, 40, 1, 80, 16).to(device)
    audio_ratio_input = torch.randn(1, 40, 1).to(device)
    image_exp_pose = torch.randn(1, 40, 70).to(device)
    # input ---- audio_mels, audio_ratio, image_exp_pose
    #   tensor [audio_mels] size: [1, 200, 1, 80, 16], min: -4.0, max: 2.590095, mean: -1.017794
    #   tensor [audio_ratio] size: [1, 200, 1], min: 0.0, max: 1.0, mean: 0.6575
    #   tensor [image_exp_pose] size: [1, 200, 70], min: -1.156697, max: 1.459776, mean: 0.023419
    # output ---- tensor [audio_exp_pose] size: [200, 70], min: -1.703708, max: 1.255959, mean: -0.02074

    with torch.no_grad():
        dummy_output = model(audio_mels_input, audio_ratio_input, image_exp_pose)
    torch_outputs = [dummy_output.cpu()]

    # 2. Export onnx model
    input_names = [ "audio_mels", "audio_ratio", "image_exp_pose"]
    output_names = [ "output" ]
    onnx_filename = "output/audio_3d_exp_pose.onnx"
    dynamic_axes = { 
        'audio_mels' : {1: 'nframes'}, 
        'audio_ratio' : {1: 'nframes'}, 
        'image_exp_pose' : {1: 'nframes'}, 
        'output' : {0: 'nframes'} 
    }  

    torch.onnx.export(model, 
        (audio_mels_input, audio_ratio_input, image_exp_pose),
        onnx_filename, 
        verbose=True, 
        input_names=input_names, 
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=16,
    )

    # 3. Check onnx model file
    onnx_model = onnx.load(onnx_filename)
    onnx.checker.check_model(onnx_model)

    # onnx_model, check = simplify(onnx_model)
    # assert check, "Simplified ONNX model could not be validated"
    passes = ["extract_constant_to_initializer", "eliminate_unused_initializer"]    
    onnx_model = onnxoptimizer.optimize(onnx_model, passes)
    onnx.save(onnx_model, onnx_filename)
    # print(onnx.helper.printable_graph(onnx_model.graph))

    # 4. Run onnx model
    if 'cuda' in device.type:
        ort_session = onnxruntime.InferenceSession(onnx_filename, providers=['CUDAExecutionProvider'])
    else:        
        ort_session = onnxruntime.InferenceSession(onnx_filename, providers=['CPUExecutionProvider'])

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # audio_mels_input, audio_ratio_input
    onnx_inputs = { input_names[0]: to_numpy(audio_mels_input), 
                    input_names[1]: to_numpy(audio_ratio_input),
                    input_names[2]: to_numpy(image_exp_pose),
                }
    onnx_outputs = ort_session.run(None, onnx_inputs)

    # 5.Compare output results
    assert len(torch_outputs) == len(onnx_outputs)
    for torch_output, onnx_output in zip(torch_outputs, onnx_outputs):
        torch.testing.assert_close(torch_output, torch.tensor(onnx_output), rtol=0.01, atol=0.01)

    todos.model.reset_device()

    print("!!!!!! Torch and ONNX Runtime output matched !!!!!!")


def export_image_3d_keypoint_onnx_model():
    import onnx
    import onnxruntime
    from onnxsim import simplify
    import onnxoptimizer

    print("Export image_3d_keypoint onnx model ...")

    # 1. Run torch model
    model, device = SAD.get_script_model()
    model = model.kpdetector_model

    B, C, H, W = 1, 3, 512, 512
    dummy_input = torch.randn(B, C, H, W).to(device)

    with torch.no_grad():
        dummy_output = model(dummy_input)
    torch_outputs = [dummy_output.cpu()]

    # 2. Export onnx model
    input_names = [ "input" ]
    output_names = [ "output" ]
    onnx_filename = "output/image_3d_keypoint.onnx"

    torch.onnx.export(model, 
        (dummy_input),
        onnx_filename, 
        verbose=False, 
        input_names=input_names, 
        output_names=output_names,
        opset_version=16,
    )

    # 3. Check onnx model file
    onnx_model = onnx.load(onnx_filename)
    onnx.checker.check_model(onnx_model)

    onnx_model, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx_model = onnxoptimizer.optimize(onnx_model)
    onnx.save(onnx_model, onnx_filename)
    # print(onnx.helper.printable_graph(onnx_model.graph))

    # 4. Run onnx model
    if 'cuda' in device.type:
        ort_session = onnxruntime.InferenceSession(onnx_filename, providers=['CUDAExecutionProvider'])
    else:        
        ort_session = onnxruntime.InferenceSession(onnx_filename, providers=['CPUExecutionProvider'])

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    onnx_inputs = { input_names[0]: to_numpy(dummy_input), 
                }
    onnx_outputs = ort_session.run(None, onnx_inputs)

    # 5.Compare output results
    assert len(torch_outputs) == len(onnx_outputs)
    for torch_output, onnx_output in zip(torch_outputs, onnx_outputs):
        torch.testing.assert_close(torch_output, torch.tensor(onnx_output), rtol=0.01, atol=0.01)

    todos.model.reset_device()

    print("!!!!!! Torch and ONNX Runtime output matched !!!!!!")


def export_audio_face_render_onnx_model():
    import onnx
    import onnxruntime
    from onnxsim import simplify
    import onnxoptimizer

    print("Export audio_face_render onnx model ...")

    # 1. Run torch model
    model, device = SAD.get_script_model()
    model = model.sadkernel_model

    B, C, H, W = 1, 3, 512, 512
    image = torch.randn(B, C, H, W).to(device) # source_kp
    audio_kp = torch.randn(B, 15, 3).to(device) # offset_kp
    image_kp = torch.randn(B, 15, 3).to(device) # source_image
    # image = torch.load("output/image.tensor").to(device)
    # audio_kp = torch.load("output/audio_kp.tensor").to(device)
    # image_kp = torch.load("output/image_kp.tensor").to(device)

    # image, audio_kp=audio_kp, image_kp=image_kp
    with torch.no_grad():
        dummy_output = model(image, audio_kp, image_kp)

    torch_outputs = [dummy_output.cpu()]

    # 2. Export onnx model
    input_names = [ "image", "audio_kp", "image_kp" ]
    output_names = [ "output" ]
    onnx_filename = "output/audio_face_render.onnx"

    torch.onnx.export(model, 
        (image, audio_kp, image_kp),
        onnx_filename, 
        verbose=False, 
        input_names=input_names, 
        output_names=output_names,
        opset_version=16,
    )

    # 3. Check onnx model file
    onnx_model = onnx.load(onnx_filename)
    onnx.checker.check_model(onnx_model)
    onnx_model, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx_model = onnxoptimizer.optimize(onnx_model)
    onnx.save(onnx_model, onnx_filename)
    print(onnx.helper.printable_graph(onnx_model.graph))

    # 4. Run onnx model
    if 'cuda' in device.type:
        ort_session = onnxruntime.InferenceSession(onnx_filename, providers=['CUDAExecutionProvider'])
    else:        
        ort_session = onnxruntime.InferenceSession(onnx_filename, providers=['CPUExecutionProvider'])

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    onnx_inputs = { input_names[0]: to_numpy(image), 
                    input_names[1]: to_numpy(audio_kp),
                    input_names[2]: to_numpy(image_kp),
                }
    onnx_outputs = ort_session.run(None, onnx_inputs)

    # 5.Compare output results
    assert len(torch_outputs) == len(onnx_outputs)
    for torch_output, onnx_output in zip(torch_outputs, onnx_outputs):
        torch.testing.assert_close(torch_output, torch.tensor(onnx_output), rtol=0.05, atol=0.05)

    todos.model.reset_device()

    print("!!!!!! Torch and ONNX Runtime output matched !!!!!!")


def export_3dmm_keypoint_map_onnx_model():
    import onnx
    import onnxruntime
    from onnxsim import simplify
    import onnxoptimizer

    print("Export 3dmm_keypoint_map onnx model ...")

    # 1. Run torch model
    model, device = SAD.get_script_model()
    model = model.mappingnet_model

    input_kp = torch.randn(1, 15, 3).to(device)
    input_3dmm = torch.randn(70, 27).to(device)

    input_kp = torch.load("/tmp/input_kp.tensor").to(device)
    input_3dmm = torch.load("/tmp/input_3dmm.tensor").to(device)

    # input ---- input_kp, input_3dmm
    # tensor [input_kp] size: [1, 15, 3], min: -0.891859, max: 0.950069, mean: 0.015366
    # tensor [input_3dmm] size: [70, 27] , min: tensor(-1.1567, device='cuda:0') , max: tensor(1.4598, device='cuda:0')
    # output ---- [fine_kp] size: [1, 15, 3]
    with torch.no_grad():
        dummy_output = model(input_kp, input_3dmm)

    torch_outputs = [dummy_output.cpu()]

    # 2. Export onnx model
    input_names = [ "input_kp", "input_3dmm"]
    output_names = [ "output" ]
    onnx_filename = "output/3dmm_keypoint_map.onnx"

    torch.onnx.export(model, 
        (input_kp, input_3dmm),
        onnx_filename, 
        verbose=True, 
        input_names=input_names, 
        output_names=output_names,
        opset_version=16,
    )

    # 3. Check onnx model file
    onnx_model = onnx.load(onnx_filename)
    onnx.checker.check_model(onnx_model)
    # onnx_model, check = simplify(onnx_model)
    # assert check, "Simplified ONNX model could not be validated"
    onnx_model = onnxoptimizer.optimize(onnx_model)
    onnx.save(onnx_model, onnx_filename)
    print(onnx.helper.printable_graph(onnx_model.graph))

    # 4. Run onnx model
    if 'cuda' in device.type:
        ort_session = onnxruntime.InferenceSession(onnx_filename, providers=['CUDAExecutionProvider'])
    else:        
        ort_session = onnxruntime.InferenceSession(onnx_filename, providers=['CPUExecutionProvider'])

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    onnx_inputs = { input_names[0]: to_numpy(input_kp), 
                    input_names[1]: to_numpy(input_3dmm),
                }

    onnx_outputs = ort_session.run(None, onnx_inputs)

    # 5.Compare output results
    assert len(torch_outputs) == len(onnx_outputs)
    for torch_output, onnx_output in zip(torch_outputs, onnx_outputs):
        torch.testing.assert_close(torch_output, torch.tensor(onnx_output), rtol=0.05, atol=0.05)

    todos.model.reset_device()

    print("!!!!!! Torch and ONNX Runtime output matched !!!!!!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Smoke Test')
    parser.add_argument('-s', '--shape_test', action="store_true", help="test shape")
    parser.add_argument('-b', '--bench_mark', action="store_true", help="test benchmark")
    parser.add_argument('-e', '--export_onnx', action="store_true", help="txport onnx model")
    args = parser.parse_args()

    if args.shape_test:
        test_input_shape()
    if args.bench_mark:
        run_bench_mark()
    if args.export_onnx:
        # Trace mode and CPU seems OK
        # export_image_3d_exp_pose_onnx_model() # OK !!!

        export_audio_3d_exp_pose_onnx_model() # ???

        # export_image_3d_keypoint_onnx_model() # OK
        # export_audio_face_render_onnx_model() # ???
        # export_3dmm_keypoint_map_onnx_model() # OK
    
    if not (args.shape_test or args.bench_mark or args.export_onnx):
        parser.print_help()