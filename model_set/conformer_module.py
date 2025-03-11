"""
EEG Conformer

Convolutional Transformer for EEG decoding

Couple CNN and Transformer in a concise manner with amazing results
"""

import os
gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
import math
from torchsummary import summary
import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor

from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True


# Convolution module
# use conv to capture local features, instead of position embedding.
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40, channel=27):
        # self.patch_size = patch_size
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=40, kernel_size=(1, 25), stride=(1, 1)),
            nn.Conv2d(40, 40, (channel, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),           # 合并 h 和 w 维度
        )

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x

class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (emb_size // num_heads) ** 0.5
        self.queries = nn.Linear(emb_size, emb_size)
        self.keys = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(dropout)
        self.out_projection = nn.Linear(emb_size, emb_size)

    def forward(self, x, mask=None):
        b, n, _ = x.shape
        q = self.queries(x).view(b, n, self.num_heads, -1)
        k = self.keys(x).view(b, n, self.num_heads, -1)
        v = self.values(x).view(b, n, self.num_heads, -1)

        qk = torch.einsum('bnhq,bnhk->bhqk', q, k) / self.scale
        if mask is not None:
            qk.masked_fill_(~mask.unsqueeze(1), float('-inf'))

        attn = F.softmax(qk, dim=-1)
        attn = self.dropout(attn)

        weighted_avg = torch.einsum('bhqk,bnhk->bnhq', attn, v).view(b, n, -1)
        return self.out_projection(weighted_avg)

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Module):
    def __init__(self, emb_size, expansion, drop_p):
        super(FeedForwardBlock, self).__init__()
        self.linear1 = nn.Linear(emb_size, expansion * emb_size)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(drop_p)
        self.linear2 = nn.Linear(expansion * emb_size, emb_size)
        self.dropout2 = nn.Dropout(drop_p)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size, num_heads=10, drop_p=0.5, forward_expansion=4, forward_drop_p=0.5):
        super(TransformerEncoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.mha = ResidualAdd(nn.Sequential(
            MultiHeadAttention(emb_size, num_heads, drop_p),
            nn.Dropout(drop_p)
        ))
        self.norm2 = nn.LayerNorm(emb_size)
        self.ff = ResidualAdd(nn.Sequential(
            FeedForwardBlock(emb_size, forward_expansion, forward_drop_p),
            nn.Dropout(drop_p)
        ))

    def forward(self, x):
        x = self.norm1(x)
        x = self.mha(x)
        x = self.norm2(x)
        x = self.ff(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, depth, emb_size):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(emb_size) for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerModule(nn.Module):
    def __init__(self, depth, emb_size):
        super().__init__()
        self.encoder = TransformerEncoder(depth, emb_size)

    def forward(self, x):
        return self.encoder(x)


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes, cla):
        super().__init__()
        self.cla = cla

        # global average pooling
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
        self.fc = nn.Sequential(
            nn.Linear(2440, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
        )
        self.classification = nn.Linear(32, n_classes)

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)      # 张量连续化, flatten
        out = self.fc(x)
        if self.cla == True:
            out = self.classification(out)
        return out

class Conformer(nn.Module):
    def __init__(self, cla=True, emb_size=40, depth=6, n_classes=2, channel=10):
        super().__init__()
        self.patch_embedding = PatchEmbedding(emb_size, channel)
        self.encoder = TransformerModule(depth, emb_size)
        self.classification_head = ClassificationHead(emb_size, n_classes, cla)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.encoder(x)
        x = self.classification_head(x)
        return x



if __name__ == "__main__":
    model = Conformer(channel=8, n_classes=4, cla=True).cuda()
    a = torch.randn(64, 1, 8, 1000).cuda().float()
    outputs = model(a)

    summary(model, input_size=(1, 8, 1000), batch_size=64)
    print(outputs.shape)

