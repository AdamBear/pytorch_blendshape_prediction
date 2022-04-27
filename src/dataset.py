import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
from collections import OrderedDict
import re
import os
from decimal import Decimal

from torch.nn.utils.rnn import pad_sequence

# data_dir should be structured as follows:
# - speaker_id_1/
# - speaker_id_1/alignments
# - speaker_id_1/phones.txt
# - speaker_id_1/sample_id1.csv (blendshapes)
# - speaker_id_1/sample_id2.csv
# - speaker_id_2/sample_id1.csv
# ..
class VisemeAlignmentDataset(Dataset):
    def __init__(self, data_dir, preprocess_visemes, preprocess_alignments, pad_value=None):
        self.pad_value = pad_value
        self.preprocess_visemes = preprocess_visemes
        self.preprocess_alignments = preprocess_alignments
        self.inputs = []
        self.lengths = []
        self.processed = {}

        alignments_file=os.path.join(data_dir, "alignments")
        if os.path.exists(alignments_file) is not True:
            raise Exception(f"File [ alignment ] does not exist in directory {data_dir}")
        phones_file=os.path.join(data_dir, "phones.txt")
        if os.path.exists(phones_file) is not True:
            raise Exception(f"File [ phones.txt ] does not exist in directory {data_dir}")
        phones = {}
        for line in open(phones_file, "r").readlines():
            split =line.strip().split(" ")
            phones[split[1]] = split[0]
        self.alignments = {}
        with open(alignments_file,"r") as infile:
            for line in infile:
                split=line.strip().split(" ")
                # also note that alignments are keyed by SPKR_ID-UTT_ID, but viseme files are keyed by UTT_ID only, so we just remove the speaker ID here
                utt_id=split[0].split("-")[1]
                print(f"ID {utt_id} {len(split)} {list(range(1, len(split), 3))}")
                # print(split)
                self.alignments[utt_id] = [(phones[split[i]],int(split[i+1])) for i in range(1, len(split), 3)]
        
        for viseme_file in list(Path(data_dir).rglob("*.csv")):
            viseme_file = str(viseme_file)
            if re.search("[0-9]\.[0.9]", viseme_file) is None:
                
                utt_id=os.path.basename(viseme_file).replace(".csv","")
                # sometimes there won't be a phone alignment row for a given viseme file (alignment may have failed)
                # just skip these
                
                if utt_id not in self.alignments:
                    continue
                self.inputs.append((viseme_file,self.alignments[utt_id]))
        self.inputs.sort(key=lambda x: len(x[1]))
                
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
        # x_pad = torch.nn.utils.rnn.pack_padded_sequence(xs, x_lens)
        # y_pad = torch.nn.utils.rnn.pack_padded_sequence(ys, y_lens)
        return x_pad, y_pad, x_lens, y_lens, _
    

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if idx not in self.processed:
            visemes = self.preprocess_visemes(self.inputs[idx][0]).values.astype(np.float32)       
            alignments = self.preprocess_alignments(self.inputs[idx][1])
            self.processed[idx] = torch.LongTensor(alignments), torch.tensor(visemes), self.inputs[idx][1]
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
        self.inputs = []
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




