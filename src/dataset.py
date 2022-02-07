import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
from collections import OrderedDict
import math
import os
from decimal import Decimal

from torch.nn.utils.rnn import pad_sequence

# data_dir should be structured as follows:
# - speaker_id_1/
# - speaker_id_1/sample_id1.csv (blendshapes)
# - speaker_id_1/sample_id1.ctm (phonetic alignments)
# - speaker_id_1/sample_id2.csv
# - speaker_id_1/sample_id2.ctm 
# - speaker_id_2/sample_id1.csv
# - speaker_id_2/sample_id1.ctm
# ..
class VisemeAlignmentDataset(Dataset):
    def __init__(self, data_dir, preprocess_visemes, preprocess_alignments, pad_value=None):
        self.pad_value = pad_value
        self.preprocess_visemes = preprocess_visemes
        self.preprocess_alignments = preprocess_alignments
        self.visemes = []
        self.alignments = []    
        self.processed = {}
        for viseme_file in list(Path(data_dir).rglob("*.csv")):
            viseme_file = str(viseme_file)
            ctm_file = str(viseme_file).replace("csv","ctm")            
            if os.path.exists(ctm_file):
                self.visemes.append(viseme_file)
                self.alignments.append(ctm_file)
    


    def trim(self, batch):
        for x, y, _ in batch:
            x_len = len(x)
            y_len = len(y)
            if x_len != y_len:
                x_len = min(x_len, y_len)
                y_len = x_len
            yield x[:x_len], y[:y_len], x_len, y_len, _
            
    def collate(self, batch, pad_val=None):
        
        trimmed = list(self.trim(batch))
        
        xs, ys, x_lens, y_lens, _ = list(zip(*trimmed))

        x_pad = pad_sequence(xs, padding_value=pad_val, batch_first=True)
        y_pad = pad_sequence(ys, padding_value=pad_val, batch_first=True)

        return x_pad, y_pad, x_lens, y_lens, _
    

    def __len__(self):
        return len(self.visemes)

    def __getitem__(self, idx):
        if idx not in self.processed:
            visemes = self.preprocess_visemes(self.visemes[idx]).values.astype(np.float32)       
            alignments = self.preprocess_alignments(self.alignments[idx])
            self.processed[idx] = torch.IntTensor(alignments), torch.tensor(visemes), self.alignments[idx]
        return self.processed[idx]

# data_dir should be structured as follows:
# - speaker_id_1/
# - speaker_id_1/sample_id1.wav (audio)
# - speaker_id_1/sample_id1.csv (blendshapes)
# - speaker_id_1/sample_id1_pp.txt (transcript)
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




