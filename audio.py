import soundfile as sf
import librosa
import numpy as np

def pad_audio(audio, sample_rate, pad_len_in_secs):
    # left-pad the audio so we have the left context when starting at the initial viseme
    pad_len_in_samples = pad_len_in_secs * sample_rate 
    if len(audio.shape) > 1:
        audio = audio[0]
    if audio.shape[0] < pad_len_in_samples:
        audio = np.pad(audio, (0, (pad_len_in_secs * sample_rate) - audio.shape[0]), constant_values=0.001)
    elif audio.shape[0] > pad_len_in_samples:
        audio = audio[:pad_len_in_samples]
    #audio = np.hstack([np.zeros((win_length*2,)), audio])
    return audio

def load_and_pad_audio(filepath, resample_to, pad_len_in_secs):
    audio, rate = sf.read(filepath)
    audio = librosa.resample(audio, rate, resample_to)
    audio = pad_audio(audio, resample_to, pad_len_in_secs)
    return audio


