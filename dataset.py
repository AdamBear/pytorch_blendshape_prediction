import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
from collections import OrderedDict

# data_dir should be structured as follows:
# - speaker_id_1/
# - speaker_id_1/sample_id1.wav
# - speaker_id_1/sample_id1.csv
# - speaker_id_1/sample_id2.wav
# - speaker_id_1/sample_id2.wav
# - speaker_id_2/sample_id1.wav
# ..
class VisemeDataset(Dataset):
    def __init__(self, data_dir, audio_transform, viseme_transform, num_ipa_symbols):
        self.viseme_transform= viseme_transform
        self.audio_transform = audio_transform
        self.audio_files = []
        self.transcripts = []
        self.visemes = []
        self.processed = {}
        self.num_ipa_symbols = num_ipa_symbols
        for file in list(Path(data_dir).rglob("*.wav")):
            self.audio_files.append(file)
            self.transcripts.append(str(file).replace(".wav", "_pp.txt"))
            self.visemes.append(str(file).replace("wav", "csv"))

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        if idx not in self.processed:
            fft, num_samples, mask = self.audio_transform(self.audio_files[idx])
            viseme_filename = self.visemes[idx]
            visemes = torch.tensor(self.viseme_transform(viseme_filename).values.astype(np.float32))       
            ipa_indices = []
            with open(self.transcripts[idx], "r") as infile:
                indices = infile.read()[0].split("#")
                for ind in indices:
                    ipa_indices.append(int(ind))
                    ipa_indices.append(self.num_ipa_symbols)
            self.processed[idx] = fft, torch.tensor(ipa_indices), torch.tensor(mask), visemes, viseme_filename
        return self.processed[idx]

def preprocess_viseme(csv, pad_len_in_secs=None, target_framerate=None, blendshapes=None):
    csv = pd.read_csv(csv)
    
    # first, drop every nth row to reduce effective framerate    
    #csv = csv.iloc[::int(59.97 / target_framerate)]
    
    pad_len = int(pad_len_in_secs * target_framerate)
        
    if(csv.shape[0] < pad_len):
        pad = pd.DataFrame(0, index=[i for i in range(pad_len - csv.shape[0])], columns=csv.columns)
        pad.pad(inplace=True)
        csv = pd.concat([csv, pad])
    else:
        csv = csv.iloc[:pad_len]
        #print("Visemes exceeded max length, truncate?")
    columns = list(csv.columns)
    columns.remove("Timecode")
    
    return csv[blendshapes] if blendshapes is not None else csv


def collate_samples(feat_tuples):
    return feat_tuples
    padded = torch.nn.utils.rnn.pad_sequence([f[0] for f in feat_tuples], batch_first=True, padding_value=0.0)
    #mask = torch.stack([feat_tuples[i][1] for i in range(len(feat_tuples))])
    labels = torch.nn.utils.rnn.pad_sequence([f[2] for f in feat_tuples], batch_first=True, padding_value=0.0)
    viseme_filenames = [feat_tuples[i][3] for i in range(len(feat_tuples))]
    
    return padded, labels,viseme_filenames


