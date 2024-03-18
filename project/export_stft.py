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

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import onnx
import onnxruntime
from onnxsim import simplify
import onnxoptimizer

import torchaudio

import todos
import pdb


# https://github.com/NVIDIA/mellotron/blob/master/stft.py
class STFT(nn.Module):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""
    def __init__(self, filter_length=800, hop_length=200, win_length=800):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        cutoff = int((self.filter_length / 2 + 1))

        fourier_basis = np.fft.fft(np.eye(self.filter_length))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]), 
                        np.imag(fourier_basis[:cutoff, :])])
        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        fft_window = torch.hann_window(win_length)

        self.register_buffer('forward_basis', forward_basis * fft_window)

    def forward(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        # similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode='reflect')
        input_data = input_data.squeeze(1) # [1, 1, 1, 128800]--> [1, 1, 128800]

        forward_transform = F.conv1d(
            input_data,
            self.forward_basis,
            stride=self.hop_length,
            padding=0)

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part**2 + imag_part**2)

        return magnitude


class AudioSpectrogram(nn.Module):
    def __init__(self):
        super().__init__()
        self.stft = STFT(hop_length=200, win_length=800)

    def forward(self, wav):
        B = wav.shape[0]

        # Pre-emphasizes a waveform along its last dimension
        # tensor [wav] size: [200, 640], min: -1.013043, max: 1.073747, mean: -8.6e-05
        wav = torchaudio.functional.preemphasis(wav, coeff = 0.97) 
        # tensor [wav] size: [200, 640], min: -0.850947, max: 0.888253, mean: 2e-05

        # Mel Filter Bank, generates the filter bank for converting frequency bins to mel-scale bins
        # Create a frequency bin conversion matrix.
        mel_filters = torchaudio.functional.melscale_fbanks(
            401, # int(hp.n_fft // 2 + 1),
            n_mels=80, # hp.num_mels,
            f_min=55.0, #hp.fmin,
            f_max=7600.0, # hp.fmax,
            sample_rate=16000, # hp.sample_rate,
            norm="slaney",
        ).to(wav.device)

        # tensor [mel_filters] size: [401, 80], min: 0.0, max: 0.040298, mean: 0.000125

        # D = torchaudio.functional.spectrogram(
        #     wav.reshape(-1), # should 1-D data
        #     pad=0, window=torch.hann_window(800),
        #     n_fft=800, win_length=800, hop_length=200,
        #     power=1.0, normalized=False, # !!! here two items configuration is very import !!!
        # )
        # tensor [D] size: [401, 641], min: 3.3e-05, max: 38.887188, mean: 0.333159

        D = self.stft(wav.reshape(1, -1)).squeeze(0)

        S = torch.mm(mel_filters.T, D) # mel_filters.T.size() -- [80, 401]
        # tensor [S] size: [80, 641], min: 5.4e-05, max: 1.314648, mean: 0.017384

        # Amp to DB
        # min_level = math.exp(hp.min_level_db / 20.0 * math.log(10.0))
        # return 20.0 * torch.log10(torch.maximum(torch.Tensor([min_level]), x)) - hp.min_level_db
        S = torch.clamp(S, 9.9999e-06)
        S = 20.0 * torch.log10(S) - 20.0

        # Normalize
        # S = (2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value    
        # S =torch.clamp(S, -hp.max_abs_value, hp.max_abs_value)
        S = 8.0 * ((S + 100.0) / 100.0) - 4.0

        orig_mel = S.clamp(-4.0, 4.0).transpose(1, 0) # size() -- [641, 80]
        # tensor [orig_mel] size: [641, 80], min: -4.0, max: 2.590095, mean: -1.016412

        mels_list = []
        mel_step_size = 16
        for i in range(B): # B -- 200
            start_frame_num = i - 2
            start_idx = int(80 * (start_frame_num/25.0)) #hp.num_mels -- 80, hp.fps = 25.0
            end_idx = start_idx + mel_step_size
            seq = list(range(start_idx, end_idx))
            seq = [ min(max(item, 0), orig_mel.shape[0] - 1) for item in seq ]

            m = orig_mel[seq, :]
            mels_list.append(m.transpose(1, 0))

        # mels[0] size() -- [80, 16]
        mels = torch.stack(mels_list, dim=0)
        mels = mels.unsqueeze(0) # size() -- [1, 200, 80, 16]

        return mels


def export_audio_spectrogram_onnx_model():
    print("Export audio_spectrogram onnx model ...")

    # 1. Run torch model
    device = torch.device("cpu")
    model = AudioSpectrogram()
    model = model.to(device)
    model.eval()

    # torch._C._jit_set_profiling_executor(False)
    # model = torch.jit.script(model)

    wav = torch.randn(200, 640).to(device)
    with torch.no_grad():
        dummy_output = model(wav) # [1, 200, 80, 16]

    torch_outputs = [dummy_output.cpu()]

    # 2. Export onnx model
    input_names = [ "wav"]
    output_names = [ "output" ]
    onnx_filename = "output/audio_spectrogram.onnx"

    dynamic_axes = { 
        'wav' : {0: 'nframes'}, 
        'output' : {1: 'nframes'} 
    }  

    torch.onnx.export(model, 
        (wav),
        onnx_filename, 
        verbose=False, 
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
    # onnx_model = onnxoptimizer.optimize(onnx_model)
    # onnx.save(onnx_model, onnx_filename)
    # print(onnx.helper.printable_graph(onnx_model.graph))

    # 4. Run onnx model
    if 'cuda' in device.type:
        ort_session = onnxruntime.InferenceSession(onnx_filename, providers=['CUDAExecutionProvider'])
    else:        
        ort_session = onnxruntime.InferenceSession(onnx_filename, providers=['CPUExecutionProvider'])

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    onnx_inputs = { input_names[0]: to_numpy(wav), 
                }

    onnx_outputs = ort_session.run(None, onnx_inputs)

    # 5.Compare output results
    assert len(torch_outputs) == len(onnx_outputs)
    for torch_output, onnx_output in zip(torch_outputs, onnx_outputs):
        torch.testing.assert_close(torch_output, torch.tensor(onnx_output), rtol=0.05, atol=0.05)

    todos.model.reset_device()

    print("!!!!!! Torch and ONNX Runtime output matched !!!!!!")


if __name__ == "__main__":
    export_audio_spectrogram_onnx_model() # NOK for 'aten::stft'
