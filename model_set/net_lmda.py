import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary


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

        # y:(64, 24, 1, 1051)  C:24  x:(64, 24, 22, 1051)
        return y * self.C * x


class MLP(nn.Module):
    def __init__(self, feature, hidden_size):
        super(MLP, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(feature, hidden_size),
            nn.ELU(),
            # nn.Dropout(0.3),
            nn.Linear(hidden_size, 4)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x


class LMDA(nn.Module):
    """
    LMDA-Net for the paper
    parameter： 22通道，4.5s的MI数据，4分类
    """
    def __init__(self, classification, chans=22, samples=1125, num_classes=4, depth=9, kernel=75, channel_depth1=24,
                 channel_depth2=9, ave_depth=1, avepool=5):
        super(LMDA, self).__init__()
        self.classification = classification
        self.ave_depth = ave_depth
        self.channel_weight = nn.Parameter(torch.randn(depth, 1, chans), requires_grad=True)  #（9, 1, 22)
        self.relu = nn.ReLU(inplace=True)
        nn.init.xavier_uniform_(self.channel_weight.data)

        self.time_conv = nn.Sequential(
            nn.Conv2d(depth, channel_depth1, kernel_size=(1, 1), groups=1, bias=False),
            nn.BatchNorm2d(channel_depth1),
            nn.Conv2d(channel_depth1, channel_depth1, kernel_size=(1, kernel),
                      groups=channel_depth1, bias=False),
            nn.BatchNorm2d(channel_depth1),
            nn.GELU(),
        )

        self.chanel_conv = nn.Sequential(
            nn.Conv2d(channel_depth1, channel_depth2, kernel_size=(1, 1), groups=1, bias=False),
            nn.BatchNorm2d(channel_depth2),
            nn.Conv2d(channel_depth2, channel_depth2, kernel_size=(chans, 1), groups=channel_depth2, bias=False),
            nn.BatchNorm2d(channel_depth2),
            nn.GELU(),
        )

        self.norm = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 1, avepool)),
            # nn.AdaptiveAvgPool3d((9, 1, 35)),
            nn.Dropout(p=0.65),
        )

        # 定义自动填充模块
        out = torch.ones((1, 1, chans, samples))
        out = torch.einsum('bdcw, hdc->bhcw', out, self.channel_weight)
        out = self.time_conv(out)
        # out : (32, 24, 22, 1051)
        N, C, H, W = out.size()

        self.depthAttention = EEGDepthAttention(W, C, k=7)

        out = self.chanel_conv(out)
        out = self.norm(out)
        n_out_time = out.cpu().data.numpy().shape

        final_features = n_out_time[-1]*n_out_time[-2]*n_out_time[-3]       # 378
        mlp_features = 5
        hidden_feature = 64

        self.fc1 = nn.Linear(final_features, mlp_features)
        self.bn1 = nn.BatchNorm1d(mlp_features)
        self.MLP = MLP(mlp_features, hidden_feature)
        self.softmax = nn.Softmax(1)

        print('In LMDA-Net, n_out_time shape: ', n_out_time)
        self.classifier = nn.Linear(n_out_time[-1]*n_out_time[-2]*n_out_time[-3], num_classes)

    def forward(self, x):
        x = torch.einsum('bdcw, hdc->bhcw', x, self.channel_weight)  # 导联权重筛选 (32, 1, 22, 1125) -> (32, 9, 22, 1125)

        x_time = self.time_conv(x)  # batch, depth1, channel, samples_
        x_time = self.depthAttention(x_time)  # DA1

        x = self.chanel_conv(x_time)  # batch, depth2, 1, samples_
        x = self.norm(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        # x = self.bn1(x)

        if self.classification:
            x = self.MLP(x)
            x = self.softmax(x)
        return x


if __name__ == '__main__':
    model = LMDA(num_classes=4, chans=22, samples=1000, channel_depth1=24, channel_depth2=9, classification=True,
                 avepool=250//10).cuda()
    a = torch.randn(64, 1, 22, 1000).cuda().float()
    l2 = model(a)
    model_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    summary(model, input_size=(1, 22, 1000), batch_size=64)
    print(l2.shape)
