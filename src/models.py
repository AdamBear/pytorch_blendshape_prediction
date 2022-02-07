import torch
from torch import nn

class SeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SeparableConv1d, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, 
                               groups=in_channels, bias=bias, padding=1)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 
                               kernel_size=1, bias=bias)
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
    
class Conv1dModel(nn.Module):
    def __init__(self, seq_length=633, n_ffts=2457, num_viseme=4, ks=256):
        super(Conv1dModel, self).__init__()
        self.conv = nn.Sequential(
            SeparableConv1d(seq_length,seq_length,ks),
            nn.ReLU(),
            SeparableConv1d(seq_length,seq_length,ks),
            nn.ReLU(),
        )
        self.attention = nn.MultiheadAttention(1951, 1, batch_first=True)
        #self.linear_relu_stack2 = nn.Sequential(
        #    SeparableConv1d(seq_length,seq_length,ks),
        #    #nn.Linear(seq_length, seq_length),
        #    nn.ReLU(),7
        #)
        self.linear_out = nn.Linear(1951, 11)

    def forward(self, x):
        o1 = self.conv(x)
        attn_output, attn_output_weights = self.attention(o1, o1, o1)
        attn_output = attn_output.tile((4,1,1,1)).transpose(0,1)
        o1 = self.linear_out(attn_output)        
        return o1

class BiLSTMModel(nn.Module):
    def __init__(self, input_dim=None, hidden_size=512, num_visemes=4):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(input_dim, hidden_size, 1, bidirectional=False)
        self.attention = nn.MultiheadAttention(hidden_size, 1, batch_first=True)
        self.output_linear = nn.ModuleList([torch.nn.Linear(hidden_size, 1) for v in range(num_visemes)])

    def forward(self, x):
        out_f, _ = self.lstm(x)
        attn_output, attn_output_weights = self.attention(out_f, out_f, out_f)
        #return self.output_linear(attn_output)
        return torch.stack([v(attn_output) for v in self.output_linear],dim=2)

class SeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SeparableConv1d, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size,
                               groups=in_channels, bias=bias, padding=1)
        self.pointwise = nn.Conv1d(in_channels, out_channels,
                               kernel_size=1, bias=bias)
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
    
class Conv1dModel(nn.Module):
    def __init__(self, 
                 seq_length=None, 
                 input_dim=None, 
                 text_dim=None, 
                 text_embedding_dim=512,
                 text_padding_idx=None,
                 num_visemes=4, 
                 ks=128):
        super(Conv1dModel, self).__init__()
        #self.text_padding_idx = text_padding_idx;
                
        self.conv1 = nn.Sequential(
            #nn.Linear(1200,1075),
            SeparableConv1d(seq_length,seq_length,ks),
            nn.ReLU(),
            nn.MaxPool1d(64, stride=4)
            #nn.Linear(1075,1075),
            #nn.ReLU(),
        )
        #self.conv2 =  nn.Sequential(
        #    SeparableConv1d(seq_length,seq_length,ks),
        #    nn.ReLU(),
        #)
        self.text_embedding = nn.Sequential(
                nn.Embedding(text_dim,text_embedding_dim),
                nn.ReLU(),
                nn.Linear(text_embedding_dim, 253),
                nn.ReLU(),
        )
        self.text_attention = nn.MultiheadAttention(253, 1, batch_first=True)
        #self.dropout = torch.nn.Dropout(p=0.1)
        #self.audio_attention = nn.MultiheadAttention(1951, 1, batch_first=True)
        self.linear_out = nn.Sequential(
            nn.ReLU(),
            nn.Linear(253, 253),
            nn.ReLU(),
            nn.Linear(253, num_visemes))            
        
    def forward(self, audio_feats, text_feats):
        
        text_out = self.text_embedding(text_feats)        
        
        audio_out = self.conv1(audio_feats)
       
        #audio_out = self.conv2(audio_out)

        #text_out, text_out_attn_weights = self.text_attention(text_out,text_out,text_out)

        out = text_out + audio_out
        #out = audio_out
        
        #print(out.size())
        #out = self.dropout(out)
        #eturn torch.squeeze(torch.stack([v(out) for v in self.linear_out],dim=3),dim=2)
        return self.linear_out(out)
class BiLSTMModel(nn.Module):
    def __init__(self, input_dim=None, hidden_size=512, num_visemes=4):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(input_dim, hidden_size, 1, bidirectional=True)
        self.attention = nn.MultiheadAttention(hidden_size*2, 1, batch_first=True)
        self.output_linear = torch.nn.Linear(hidden_size*2, num_visemes)

    def forward(self, x, t):
        out_f, _ = self.lstm(x)
        attn_output, attn_output_weights = self.attention(out_f, out_f, out_f)
        return self.output_linear(attn_output)
        #return self.output_linear],dim=2),dim=3) 
