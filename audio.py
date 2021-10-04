import soundfile as sf
import librosa
import numpy as np
import torch
import scipy
from scipy import fft
import functools
from torch import nn
import math
import torchaudio 

def pad_audio(audio, sample_rate, pad_len_in_secs):
    # left-pad the audio so we have the left context when starting at the initial viseme
    pad_len_in_samples = pad_len_in_secs * sample_rate 
    if len(audio.shape) > 1:
        audio = audio[0]
    padded = 0
    if audio.shape[0] < pad_len_in_samples:
        padded = (pad_len_in_secs * sample_rate) - audio.shape[0]
        audio = np.pad(audio, (0, padded), constant_values=0.000)
    elif audio.shape[0] > pad_len_in_samples:
        audio = audio[:pad_len_in_samples]
    #audio = np.hstack([np.zeros((win_length*2,)), audio])
    return audio, padded

def load_and_pad_audio(filepath, resample_to, pad_len_in_secs):
    audio, rate = sf.read(filepath)#, dtype="int16")
    if rate != resample_to:
        raise Error("Sample rate mismatch")
    audio = librosa.resample(audio, rate, resample_to)
    return pad_audio(audio, resample_to, pad_len_in_secs)

def coeffs_to_windows(coeffs, config):
    output = np.zeros((config["seqLength"], config["stftFramesPerWindow"] *
        config["numMels"]))
   
    hop = int(config["stftFramesPerWindow"]/2)
    
    frameIdx = 0;
    
    for i in range(config["seqLength"]):
        
        startIdx = max(frameIdx-hop, 0)
        endIdx = min(frameIdx+hop, coeffs.shape[0])
        window = coeffs[startIdx:endIdx,:]    

        if frameIdx - hop < 0:
            window = np.pad(window, [((hop*2) - window.shape[0], 0), (0,0)], constant_values=coeffs[0,0])
        elif frameIdx + hop > coeffs.shape[0]:
            window = np.pad(window, [(0,(hop*2) - window.shape[0]),(0,0)], constant_values=coeffs[0,0])
        #print(window.shape)
        output[i] = np.reshape(window, config["stftFramesPerWindow"]*config["numMels"])
        frameIdx += hop
    return output
























#
#def kaldi_mels(filepath, 
#               config=None):
#    audio, mask = load_and_pad_audio(filepath, config["sampleRate"], config["paddedAudioLength"])
#    audio = torch.unsqueeze(torch.tensor(audio, dtype=torch.float32), dim=0)
#
#    fbank_framelength = int(1000 * config["fftSize"] / config["sampleRate"])
#    fbank_frameshift =fbank_framelength / 2 #1000 *config["hopSize"] / config["sampleRate"]
#    
#    output = torchaudio.compliance.kaldi.fbank(audio,
#                                  frame_length=fbank_framelength],
#                                  frame_shift =fbank_frameshift,
#                                  num_mel_bins=config["numMels"],
#                                  high_freq=config["fmax"], 
#                                  low_freq=config["fmin"], 
#                                  sample_frequency=config["sampleRate"],
#                                  use_energy=False,
#                                  htk_compat=False,
#                                  raw_energy=False,
#                                  snip_edges=True,
#                                  vtln_low=100, vtln_high=-500,
#                                  use_log_fbank=True,
#                                  use_power=False,
#                                  energy_floor=0.0,
#                                  window_type='povey'
#                                 )
#    #print(f"output shape {output.size()} and num_frames {num_frames}, stft frames perindow {config['stftFramesPerWindow']}")
#    output = coeffs_to_windows(output, config)
#    return torch.tensor(output,dtype=torch.float32), mask
