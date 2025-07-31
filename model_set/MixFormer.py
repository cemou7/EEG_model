from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import argparse
from math import sqrt
from torchsummary import summary
import sys
sys.path.append("/home/work3/wkh/CL-Model")


class EEGDepthAttention(nn.Module):
    """
    Build EEG Depth Attention module.
    C: num of channels
    W: num of time samples
    k: learnable kernel size
    """
    def __init__(self, W, C, k=7):
        super(EEGDepthAttention, self).__init__()
        self.C = C
        # 自适应平均池化层：将输入池化成一个高度为 1，宽度为 W 的张量
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, W))
        # 卷积核大小 (7, 1)，填充高度 (k // 2) 使输入输出保持一致
        self.conv = nn.Conv2d(1, 1, kernel_size=(k, 1), padding=(k // 2, 0), bias=True)  # original kernel k
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, x):
        x_pool = self.adaptive_pool(x)
        # 交换 channel_depth1 和 eeg_channel（高度） 列
        x_transpose = x_pool.transpose(-2, -3)
        y = self.conv(x_transpose)

        # 对通道 channel 分类
        y = self.softmax(y)
        y = y.transpose(-2, -3)

        return y * self.C * x


class ClassificationHead(nn.Module):
    def __init__(self, emb_size, n_classes):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(emb_size, 64),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)      # 张量连续化, flatten
        out = self.fc(x)
        return self.softmax(out)


class MixFormer(nn.Module):
    def __init__(self, cla):
        super(MixFormer, self).__init__()
        args_PatchTST = set_parser()
        self.PatchTST = PatchTST(args_PatchTST).cuda()

        # self.PatchTST.load_state_dict(torch.load('model/PatchTST_A03_checkpoint.pth'))
        self.PatchTST.load_state_dict(torch.load('model/PatchTST_A03_checkpoint_model_mse_less_than_1.pth'))
        
        self.layer_norm = nn.LayerNorm(500)

        self.channel_weight = nn.Parameter(torch.randn(9, 1, 22), requires_grad=True)
        self.relu = nn.ReLU(inplace=True)
        nn.init.xavier_uniform_(self.channel_weight.data)

        self.time_conv = nn.Sequential(
            nn.Conv2d(9, 24, kernel_size=(1, 1), groups=1, bias=False),
            nn.BatchNorm2d(24),
            nn.Conv2d(24, 24, kernel_size=(1, 75),
                      groups=24, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
        )

        self.chanel_conv = nn.Sequential(
            nn.Conv2d(24, 9, kernel_size=(1, 1), groups=1, bias=False),
            nn.BatchNorm2d(9),
            nn.Conv2d(9, 9, kernel_size=(22, 1), groups=9, bias=False),
            nn.BatchNorm2d(9),
            nn.GELU(),
        )

        self.norm = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 1, 25)),
            nn.Dropout(p=0.3),
        )

        # 定义自动填充模块
        out = torch.ones((1, 1, 22, 500))
        out = torch.einsum('bdcw, hdc->bhcw', out, self.channel_weight)
        out = self.time_conv(out)
        N, C, H, W = out.size()

        self.depthAttention = EEGDepthAttention(W, C, k=7)
        out = self.chanel_conv(out)
        out = self.norm(out)
        n_out_time = out.cpu().data.numpy().shape

        final_features = n_out_time[-1] * n_out_time[-2] * n_out_time[-3]
        self.fc = nn.Linear(final_features, 4)
        self.softmax = nn.Softmax(1)

    def forward(self, x):           # input:[64, 22, 1000]
        x = x.permute(0, 2, 1)
        front, rear = torch.split(x, 500, dim=1)    # [64, 500, 22]
        pred = self.PatchTST(front)     # x:[64, 500, 22]  pred:[64, 500, 22]

        # x = pred
        x = pred - rear                 # 将原始信号的后半段也加入训练
        x = x.permute(0, 2, 1)
        x = self.layer_norm(x)

        # baseline
        # x = rear.permute(0, 2, 1)    # [64, 22, 500]

        x = x.unsqueeze(1)           # [64, 1, 22, 500]
        x = torch.einsum('bdcw, hdc->bhcw', x, self.channel_weight)  # 导联权重筛选 (64, 1, 22, 500) -> (64, 9, 22, 500)
        x_time = self.time_conv(x)
        x_time = self.depthAttention(x_time)
        x = self.chanel_conv(x_time)
        x = self.norm(x)
        embed = torch.flatten(x, 1)

        x = self.fc(embed)

        x = self.softmax(x)
        return embed, x                    # 153, 4


if __name__ == '__main__':
    args = set_parser()
    model = MixFormer(cla=False).cuda()

    x = torch.randn(64, 22, 1000).cuda().float()
    embed, outputs = model(x)

    # summary(model, input_size=(22, 1000), batch_size=32)
    print(model)
    # for name, module in model.named_modules():
    #     print(name, module)
    print(embed.shape, outputs.shape)

