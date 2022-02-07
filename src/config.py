from collections import OrderedDict
import pandas as pd
import math
import json

class VisemeModelConfiguration:

    def __init__(self, batch_size=10, model_name="bilstm"):
        self.batch_size = batch_size
        mappings = OrderedDict()
        mappings["MouthClose"] = "A37_Mouth_Close"
        mappings["MouthFunnel"] = "A29_Mouth_Funnel"
        mappings["MouthPucker"] = "A30_Mouth_Pucker"
        #mappings["JawOpen"] = "Mouth_Open"
        mappings["JawOpen"] = "Merged_Open_Mouth"
        mappings["EyeBlinkLeft"] = "Eye_Blink_L"
        mappings["EyeBlinkRight"] = "Eye_Blink_R"
        mappings["EyeSquintLeft"] = "Eye_Squint_L"
        mappings["EyeSquintRight"] = "Eye_Squint_R"
        mappings["BrowDownLeft"] = "Brow_Drop_L"
        mappings["BrowDownRight"] = "Brow_Drop_R"
        mappings["MouthUpperUpLeft"] = "Mouth_Snarl_Upper_L"
        mappings["MouthUpperUpRight"] = "Mouth_Snarl_Upper_R"
        mappings["MouthLowerDownLeft"] = "Mouth_Snarl_Lower_L"
        mappings["MouthLowerDownRight"] = "Mouth_Snarl_Lower_R"
        mappings["MouthRollUpper"] = "Mouth_Top_Lip_Under" # not exact
        self.mappings = mappings

        #mappings["CheekBlow"] = "Cheek_Blow_L", "Cheek_Blow_R"
        #Nose_Scrunch = Nose_Sneer_L + Nose_Sneer_R
        
        # config for the input blendshapes
        source_config = {
            # source framerate for raw viseme label input
            "framerate":59.97,
            # we need to map the blendshapes from the incoming CSV from LiveLinkFace to CC3 blendshapes
            # unfortunately not 1-to-1, but seems to work well enough
            "mappings":mappings
        }
        self.source_config = source_config
         
        # config for the viseme model
        model_config = {}
        # the filename for the trained model that we will export below
        self.model_name = model_name
        model_config["modelPath"] = self.model_name + ".tflite"
        # the blendshapes that will be predicted
        # we also need to export this so the gltf animator can match the model outputs indices to morph target indices
        self.sourceKeys = ["MouthClose", "MouthFunnel", "MouthPucker", "JawOpen"] # ["JawOpen"] #list(mappings.keys()) # 
        model_config["targetNames"] = [mappings[n] for n in self.sourceKeys] #["A37_Mouth_Close", "A29_Mouth_Funnel", "A30_Mouth_Pucker", "Merged_Open_Mouth"] #list(mappings.values()) 
        
        # actual framerate to use for viseme labels. 
        # raw labels will be resampled/transformed (either averaged or simply dropped).
        model_config["frameRate"] = source_config["framerate"] / 14.99
        # the sample rate that audio will be resampled to
        model_config["sampleRate"] = 22050
 
        # all audio will be padded to the following size
        model_config["paddedAudioLength"] = 5
       
        # each (viseme) frame will be 1/frameRate seconds in length
        model_config["frameLength"] = (1 / model_config["frameRate"]) * model_config["sampleRate"]

        # this will be used to calculate the sequence length
        model_config["seqLength"] = int((model_config["paddedAudioLength"] * model_config["sampleRate"]) / model_config["frameLength"])

        # the raw input for each viseme frame will be an audio window of size X 
        # the middle sample of the viseme frame is aligned with the middle sample of the audio window
        # this means, at the nominal "anchor sample" of the viseme frame, there will be 
        # X/2 samples to the left and X/2 samples to the right
        model_config["windowLength"] = int(2 * model_config["sampleRate"]) 
        
        # this raw audio input will then be transformed into a number of STFT frames/coefficients
        # each viseme frame will have this number of STFT frames, which will be the actual input at each timestep
        # Since audio windows overlap, we won't want to waste cycles repeatedly computing the STFT across the whole audio sequence
        # So in practice, we pre-calculate the STFT for the whole sequence, then just sub-sample the coefficients at each timestep
        # when assigning STFT frames, the hop length will then just be half this value
        model_config["stftFramesPerWindow"] = 52
        
        model_config["hopSize"] = int(model_config["windowLength"] // model_config["stftFramesPerWindow"])
        model_config["numMels"] = 10
        model_config["fmin"] = 100
        model_config["fmax"] = 8000
        model_config["fftSize"] = 512
        self.model_config = model_config

        self.text_pad_len=model_config["seqLength"]

    def save(self, path):
        with open(path, "w",encoding="utf-8") as outfile:
            json.dump(self.model_config, outfile)
#process_audio = None
#
#if USE_KALDI_FBANK:
#    
#elif USE_TFTTS_CONFIG:
#    process_audio = functools.partial(tftts_mels, 
#                                          config=model_config)
#else:
#    num_mels=39
#    process_audio = functools.partial(stft, model_config);
#        #viseme_frame_len_in_samples=viseme_frame_len_in_samples, # this refers to the size of the viseme/audio window,
#        #audio_window_in_samples=audio_window_in_samples, # TODO - update these
#        #stft_frames_per_window=stft_frames_per_window,
#        #resample_to=resample_to, 
#        #pad_len_in_secs=pad_len_in_secs,
#        #n_mels=num_mels)

# load the TTS config so we can match the same parameters 
# this may override some of the parameters set above
#USE_TFTTS_CONFIG=False
#
#USE_KALDI_FBANK=True
#
#
#if USE_TFTTS_CONFIG:
#    with open("/mnt/hdd_2tb/home/hydroxide/projects/TensorFlowTTS/preprocess/baker_preprocess.yaml", "r") as f:
#        tftts_config = yaml.safe_load(f)
#        model_config["fftSize"] = tftts_config["fft_size"]
#        model_config["numMels"] = tftts_config["num_mels"]       
#        if "window_length" in tftts_config:
#            assert(tftts_config["window_length"] == model_config["windowLength"])
#        
#        model_config["window"] = tftts_config["window"]
#        model_config["fmin"] = tftts_config["fmin"]
#        model_config["fmax"] = tftts_config["fmax"]
#        assert(tftts_config["sampling_rate"] == model_config["sampleRate"])
#        
#        #config={"num_mels":80, "sampling_rate":22050,"fmin":80,"fmax":6000, "window":"hann", "fft_size":512, "hop_size":hop_size})
#
