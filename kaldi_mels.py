import pandas as pd
import math

import numpy as np
import soundfile as sf
import scipy
from scipy import fft
import torch
import functools
from torch import nn
import librosa
import math
import torchaudio 
import math
import yaml
from tensorflow_tts.inference import AutoConfig
import json
from audio import load_and_pad_audio, coeffs_to_windows

def kaldi_mels(filepath,
               config=None):
    audio, mask = load_and_pad_audio(filepath, config["sampleRate"], config["paddedAudioLength"])
    audio = torch.unsqueeze(torch.tensor(audio, dtype=torch.float32), dim=0)

    fbank_framelength = int(1000 * config["fftSize"] / config["sampleRate"])
    fbank_frameshift = int(fbank_framelength / 2) #1000 *config["hopSize"] / config["sampleRate"]

       
    numFrames = int(audio.shape[0] / config["sampleRate"] / fbank_frameshift)

    output = torchaudio.compliance.kaldi.fbank(audio,
                                  frame_length=fbank_framelength,
                                  frame_shift =fbank_frameshift,
                                  num_mel_bins=config["numMels"],
                                  high_freq=config["fmax"],
                                  low_freq=config["fmin"],
                                  sample_frequency=config["sampleRate"],
                                  use_energy=False,
                                  htk_compat=False,
                                  raw_energy=False,
                                  snip_edges=False,
                                  vtln_low=100, vtln_high=-500,
                                  use_log_fbank=True,
                                  use_power=False,
                                  energy_floor=0.0,
                                  window_type='povey'
                                 )
#    print(f"fbank framelength {fbank_framelength} {fbank_frameshift} {numFrames}")
  
 #   print(output.shape)
    #print(f"output shape {output.size()} and num_frames {num_frames}, stft frames perindow {config['stftFramesPerWindow']}")
    output = coeffs_to_windows(output, config)
    return torch.tensor(output,dtype=torch.float32), mask

