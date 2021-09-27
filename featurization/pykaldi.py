
from kaldi.feat.fbank import Fbank, FbankOptions, FbankComputer
from kaldi.feat.window import extract_window, FeatureWindowFunction, FrameExtractionOptions, num_frames
from kaldi.matrix import Vector

#audio = load_and_pad_audio("friend_join2.wav", 22050, 10)
audio, rate = sf.read("original.wav")
#audio = librosa.resample(audio, rate, 22050)

frame_opts = FrameExtractionOptions()
frame_opts.samp_freq = 44100
#frame_opts.frame_shift_ms = 500 / 33
#frame_opts.dither = 0.0
#frame_opts.frame_length_ms = 500
frame_opts.window_type = "hanning"
#frame_opts.allow_downsample = True


opts = FbankOptions()
opts.use_energy = False;
opts.energy_floor = 0;
opts.raw_energy = False;
opts.htk_compat = False;
opts.use_log_fbank = True;
opts.use_power = False;
opts.frame_opts= frame_opts
opts.mel_opts.num_bins = 80
opts.mel_opts.low_freq = 80.0
opts.mel_opts.high_freq = 6000.0
computer = FbankComputer(opts)

wav_vector = Vector(audio)
window = Vector(44100)

wf = FeatureWindowFunction.from_options(opts.frame_opts)
nf = num_frames(audio.shape[0], frame_opts)
print(f"{nf} frames available")
print(extract_window(0, wav_vector,0,frame_opts,wf,window))
#feature = Vector(80)

#for i in range(10):    

#extract_window(0, wav_vector,0,opts.frame_opts,FeatureWindowFunction.from_options(opts.frame_opts),window);


    #computer.compute(0.0, 1.0, window, feature)
    
    #feats_data = feats.numpy()
    #feats_data.shape


#int numFrames = NumFrames(length, opts.frame_opts);

# vector<float> frames(numFrames);

# for(int i = 0; i < numFrames; i++) {
# Vector<BaseFloat> window;
# ExtractWindow(0, waveform,i,opts.frame_opts,wf,&window);
# Vector<BaseFloat>* feature = new Vector<BaseFloat>();
# feature->Resize(predictor->config.numMels);
# /*        int d1 =        window.Dim();
# int d2 = opts.frame_opts.PaddedWindowSize();
# int d3 = feature->Dim();
# int d4 = computer.Dim(); */
# computer.Compute(0.0f, 1.0f, &window, feature);
# float* data = feature->Data();
# for(int j = 0; j < predictor->config.numMels; j++)
#   frames.push_back(data[j]);
# }


