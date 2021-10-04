import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
from collections import OrderedDict
import math

# data_dir should be structured as follows:
# - speaker_id_1/
# - speaker_id_1/sample_id1.wav
# - speaker_id_1/sample_id1.csv
# - speaker_id_1/sample_id2.wav
# - speaker_id_1/sample_id2.wav
# - speaker_id_2/sample_id1.wav
# ..
class VisemeDataset(Dataset):
    def __init__(self, data_dir, audio_transform, viseme_transform, num_ipa_symbols, text_pad_len):
        self.space = num_ipa_symbols
        self.start_token = num_ipa_symbols+1
        self.end_token = num_ipa_symbols+2
        self.pad_token = num_ipa_symbols+3
        self.num_ipa_symbols = num_ipa_symbols+4

        self.viseme_transform= viseme_transform
        self.audio_transform = audio_transform
        self.text_pad_len = text_pad_len
        self.audio_files = []
        self.transcripts = []
        self.visemes = []
        self.processed = {}
        for file in list(Path(data_dir).rglob("*.wav")):
            self.audio_files.append(file)
            self.transcripts.append(str(file).replace(".wav", "_pp.txt"))
            self.visemes.append(str(file).replace("wav", "csv"))

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        if idx not in self.processed:
            feats, num_padded = self.audio_transform(self.audio_files[idx])
            viseme_filename = self.visemes[idx]
            visemes = torch.tensor(self.viseme_transform(viseme_filename).values.astype(np.float32))       
            with open(self.transcripts[idx], "r") as infile:
                lines = infile.readlines()
                ipa_indices = [int(x) for x in lines[0].strip().split(" ")]
                while len(ipa_indices) < self.text_pad_len:
                    ipa_indices.append(self.pad_token)
                num_indices = len(ipa_indices)
                
                if num_indices > self.text_pad_len:
                    #print(f"Warning: text length {num_indices} exceeded pad length {self.text_pad_len}, trimming. This may lead to data loss.")
                    ipa_indices = ipa_indices[:self.text_pad_len]
                self.processed[idx] = feats, torch.tensor(ipa_indices), num_padded, visemes, viseme_filename
        return self.processed[idx]
def preprocess_viseme(path,
                      pad_len_in_secs=None,
                      collapse_factor=None,
                      blendshapes=None,
                      source_framerate=None):

    csv = pd.read_csv(path)

    # remove the Timecode column because we don't need it
    columns = list(csv.columns)
    columns.remove("Timecode")
    if blendshapes is not None:
        columns = [c for c in columns if c in blendshapes]
    csv = csv[columns]
    csv.reset_index()

    # pad the visemes to the intended length, using the first row as our base
    pad_len = int(pad_len_in_secs * source_framerate)
    if csv.shape[0] < pad_len:
        pad_indices = pad_len - csv.shape[0]
        csv = csv.append([csv.head(1)] * (pad_len - csv.shape[0]),ignore_index=True)
        csv[pad_indices:] = 0
    else:
        csv = csv.iloc[:pad_len]
        #print("Visemes exceeded max length, truncate?")
    
    # reduce the framerate by taking the mean of every X frames
    if collapse_factor is not None:
        i = 0
        while i < csv.shape[0]:
            csv.iloc[i] = csv.iloc[i:i+collapse_factor].mean()
            i += collapse_factor
        csv = csv.iloc[::collapse_factor]
    return csv 

def collate_samples(feat_tuples):
    return feat_tuples
    padded = torch.nn.utils.rnn.pad_sequence([f[0] for f in feat_tuples], batch_first=True, padding_value=0.0)
    #mask = torch.stack([feat_tuples[i][1] for i in range(len(feat_tuples))])
    labels = torch.nn.utils.rnn.pad_sequence([f[2] for f in feat_tuples], batch_first=True, padding_value=0.0)
    viseme_filenames = [feat_tuples[i][3] for i in range(len(feat_tuples))]
    
    return padded, labels,viseme_filenames


