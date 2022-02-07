# featurize audio in exactly the same way as TensorflowTTS
# this enables mels from the synthesis step to be reused for viseme prediction
def tftts_mels(filepath,
               config=None):

    audio, _ = load_and_pad_audio(filepath, config["sampleRate"], config["paddedAudioLength"])
        
    # this is (mostly) copied verbatim from https://github.com/TensorSpeech/TensorFlowTTS/blob/master/tensorflow_tts/bin/preprocess.py @ 4a7d584
    # except the hop length must be the size of the window (in samples) divided by the number of STFT frames per window
    assert(config["hopSize"] == int(config["windowLength"] // config["stftFramesPerWindow"]))
    
    D = librosa.stft(
        audio,
        n_fft=1024, #config["fftSize"],
        hop_length=50,
        win_length=config["fftSize"],
        #config["windowLength"] if "windowLength" in config else config["fftSize"],
        window=config["window"],
        #pad_mode="reflect",
    )
    
    S, _ = librosa.magphase(D)  # (#bins, #frames)
    fmin = 0 if config["fmin"] is None else config["fmin"]
    fmax = sampling_rate // 2 if config["fmax"] is None else config["fmax"]
    mel_basis = librosa.filters.mel(
        sr=config["sampleRate"],
        n_fft=1024,#config["fftSize"],
        n_mels=config["numMels"],
        fmin=fmin,
        fmax=fmax,
    )
    log_mels = np.log10(np.maximum(np.dot(mel_basis, S), 1e-10)).T  # (#frames, #bins)

    # coeffs = fft.dct(log_mels.T) <-- where did this come from?

    #coeffs = log_mels.T
    coeffs = log_mels
    output = coeffs_to_windows(coeffs, config)
    return torch.tensor(output, dtype=torch.float32), audio.shape[0]
