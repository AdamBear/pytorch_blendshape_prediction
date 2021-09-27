def stft(filepath, 
         frame_len, 
         window_len, 
         stft_frames_per_window,
         resample_to, 
         pad_len_in_secs, 
         n_mels):
    
    audio = load_and_pad_audio(filepath, pad_len_in_secs, resample_to)
    #wdw = np.hanning(audio_window_in_samples)
    
    num_frames = audio.shape[0] // frame_len
    # TODO - mask for padding 
    # [1] * actual_seq_length + [0] * (padded_seq_length - actual_seq_length)
    
    # take the STFT of the entire audio file 
    # with a window size equivalent to the audio_window_in_samples / audio_bins_per_window
    n_fft = int(window_len / stft_frames_per_window)

    transformed = librosa.stft(audio, 
                               n_fft=n_fft,
                               win_length=n_fft,
                               hop_length=n_fft)
    
    melfb = librosa.filters.mel(resample_to, n_fft, n_mels=n_mels)    
    mels = np.dot(melfb, np.abs(transformed))
    
    log_mels = np.log(mels)
    
    output = mels_to_mfccs(log_mels, padded_seq_length, stft_frames_per_window, n_mels)

    return output, audio.shape[0], 



