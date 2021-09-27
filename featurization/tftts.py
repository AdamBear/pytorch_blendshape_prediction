# featurize audio in exactly the same way as TensorflowTTS
# this enables mels from the synthesis step to be reused for viseme prediction
def tftts_mels(filepath, 
               config=None):
    
    audio = load_and_pad_audio(filepath, config["sampleRate"], config["paddedAudioLength"])
     
    # this is (mostly) copied verbatim from https://github.com/TensorSpeech/TensorFlowTTS/blob/master/tensorflow_tts/bin/preprocess.py @ 4a7d584 
    # except the hop length must be the size of the window (in samples) divided by the number of STFT frames per window
    assert(config["hopSize"] == int(config["windowLength"] // config["stftFramesPerWindow"]))
    
    D = librosa.stft(
        audio,
        n_fft=config["fftSize"],
        hop_length=config["hopSize"],
        win_length=config["fftSize"],
        #config["windowLength"] if "windowLength" in config else config["fftSize"],
        window=config["window"],
        pad_mode="reflect",
    )
    
    S, _ = librosa.magphase(D)  # (#bins, #frames)
    fmin = 0 if config["fmin"] is None else config["fmin"]
    fmax = sampling_rate // 2 if config["fmax"] is None else config["fmax"]
    mel_basis = librosa.filters.mel(
        sr=config["sampleRate"],
        n_fft=config["fftSize"],
        n_mels=config["numMels"],
        fmin=fmin,
        fmax=fmax,
    )
    log_mels = np.log10(np.maximum(np.dot(mel_basis, S), 1e-10)).T  # (#frames, #bins)
    
    # coeffs = fft.dct(log_mels.T) <-- where did this come from?
    
    coeffs = log_mels.T
    
    num_frames = (config["sampleRate"] * config["paddedAudioLength"]) // config["frameLength"]

    output = coeffs_to_windows(coeffs, num_frames,  config["stftFramesPerWindow"], config["numMels"])

    return output, audio.shape[0], [1] * log_mels.shape[0]


