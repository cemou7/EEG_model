"""
EEG Conformer 

Convolutional Transformer for EEG decoding

Couple CNN and Tr  ansformer in a concise manner with amazing results
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




class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):       # 40, 10, 0.5
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)   # 得到注意力权重张量 att
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):    # 40, 4, 0.5
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


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
        # 构造 depth 个 TransformerEncoderBlock 实例
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


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
            # nn.Linear(440, 256),
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


class Conformer(nn.Sequential):
    def __init__(self, cla=True, emb_size=40, depth=6, n_classes=2, channel=10, **kwargs):
        super().__init__(

            PatchEmbedding(emb_size, channel),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes, cla)
        )


if __name__ == "__main__":
    model = Conformer(channel=22, n_classes=4, cla=True).cuda()
    a = torch.randn(64, 1, 22, 1000).cuda().float()
    outputs = model(a)

    summary(model, input_size=(1, 22, 1000), batch_size=64)
    print(outputs.shape)

