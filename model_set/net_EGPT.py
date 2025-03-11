
import sys
# sys.path.append("/home/work/kober/EEG-GPT/temp")
import torchsummary
from torchsummary import summary

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

import logging
from torchsummary import summary

import math
import numpy as np
import pdb

import os
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        # nn.init.constant(m.bias, 0)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)

class channel_attention(nn.Module):
    def __init__(self, sequence_num=1000, inter=30, emb_size=64):
        super(channel_attention, self).__init__()
        self.sequence_num = sequence_num
        self.inter = inter
        self.extract_sequence = int(self.sequence_num / self.inter)  # You could choose to do that for less computation

        self.query = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.LayerNorm(emb_size),  # also may introduce improvement to a certain extent
            nn.Dropout(0.3)
        )
        self.key = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            # nn.LeakyReLU(),
            nn.LayerNorm(emb_size),
            nn.Dropout(0.3)
        )

        # self.value = self.key
        self.projection = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            # nn.LeakyReLU(),
            nn.LayerNorm(emb_size),
            nn.Dropout(0.3),
        )

        self.drop_out = nn.Dropout(0)
        self.pooling = nn.AvgPool2d(kernel_size=(1, self.inter), stride=(1, self.inter))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        temp = rearrange(x, 'b o c s->b o s c')
        temp_query = rearrange(self.query(temp), 'b o s c -> b o c s')
        temp_key = rearrange(self.key(temp), 'b o s c -> b o c s')

        channel_query = self.pooling(temp_query)
        channel_key = self.pooling(temp_key)

        scaling = self.extract_sequence ** (1 / 2)

        channel_atten = torch.einsum('b o c s, b o m s -> b o c m', channel_query, channel_key) / scaling

        channel_atten_score = F.softmax(channel_atten, dim=-1)
        channel_atten_score = self.drop_out(channel_atten_score)

        out = torch.einsum('b o c s, b o c m -> b o c s', x, channel_atten_score)
        '''
        projections after or before multiplying with attention score are almost the same.
        '''
        out = rearrange(out, 'b o c s -> b o s c')
        out = self.projection(out)
        out = rearrange(out, 'b o s c -> b o c s')
        return out

class PatchEmbedding(nn.Module):
    def __init__(self, emb_size, chans=22):
        # self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(1, emb_size, (1, 51), (1, 1)),
            nn.BatchNorm2d(emb_size),
            nn.LeakyReLU(0.2),
            nn.Conv2d(emb_size, emb_size, (chans, 5), stride=(1, 5)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))

    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)

        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x, mask = None):
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class GELU(nn.Module):
    def forward(self, input):
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                emb_size,
                num_heads=10,
                drop_p=0.5,
                forward_expansion=4,
                forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        # global average pooling
        self.out_class = sum(n_classes)
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
        )
        self.classifier = nn.Sequential(
            nn.Linear(40, 1024),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(64, self.out_class)
        )

    def forward(self, x):
        out = self.clshead(x)
        out = self.classifier(out)

        return out

class SEGPTNet(nn.Sequential):
    def __init__(self, chans=[8, 22, 3], emb_size=40, depth=8, n_classes=[3, 4, 2], **kwargs):
        super().__init__()
        
        self.maxCh = max(chans)
        self.channel_attention = channel_attention(sequence_num=1000, inter=30, emb_size=self.maxCh)
        self.petchEmbedding = PatchEmbedding(emb_size=emb_size, chans=self.maxCh)
        self.TransformerEncoder = TransformerEncoder(depth=depth, emb_size=emb_size)
        self.ClassificationHead = ClassificationHead(emb_size=emb_size, n_classes=n_classes)
        
    def ChannelEncoding(self, x):
        chans = x.size(2)
        samples = x.size(3)
        dropout = nn.Dropout(p=0.1)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(chans, samples)
        div_term = torch.ones(samples).unsqueeze(0)
        position = torch.exp(torch.arange(0, chans, 2) *
                            -(math.log(10000.0) / chans)).unsqueeze(1)
        if chans % 2 == 0:
            pe[0::2, :] = torch.sin(position * div_term)
            pe[1::2, :] = torch.cos(position * div_term)
        else:
            pe[0::2, :] = torch.sin(position * div_term)
            pe[1::2, :] = torch.cos(position[1:] * div_term)
        pe = pe.unsqueeze(0).unsqueeze(0).cuda()
        x = dropout(x + pe)
        return x

    def forward(self, x):  #(B, 1, C, S)
        # channel encoder
        # x = self.ChannelEncoding(x)
        x = self.channel_attention(x)
        x = self.petchEmbedding(x)
        x = self.TransformerEncoder(x)
        
        # x = self.ClassificationHead(x)
        
        return x
    

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('gpu use:' + str(torch.cuda.is_available()))
    print('gpu count: ' + str(torch.cuda.device_count()))

    if device == 'cuda:0':
        print('device: GPU')
        print('device index:' + str(torch.cuda.current_device()))
        print('memory allocated:', torch.cuda.memory_allocated() / 1024**2, 'MB')
        print('max memory allocated:', torch.cuda.max_memory_allocated() / 1024**2, 'MB')
    else:
        print('device: CPU')

    x1 = torch.randn(32, 1, 8, 640)
    x2 = torch.randn(32, 1, 22, 1125)
    x3 = torch.randn(32, 1, 3, 1125)
    Net = SEGPTNet(chans=[22], n_classes=[4]).to(device)
    # summary(Net, input_size=((1, 8, 640), (1, 22, 1125), (1, 3, 1125)), batch_size=32)
    summary(Net, input_size=(1, 22, 1250), batch_size=32)

