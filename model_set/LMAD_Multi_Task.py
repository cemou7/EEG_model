import os
import torch
import torch.nn as nn
from torchsummary import summary

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class EEGDepthAttention(nn.Module):
    def __init__(self, W, C, k=7):
        super(EEGDepthAttention, self).__init__()
        self.C = C
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, W))
        self.conv = nn.Conv2d(1, 1, kernel_size=(k, 1), padding=(k // 2, 0), bias=True)
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, x):
        x_pool = self.adaptive_pool(x)
        x_transpose = x_pool.transpose(-2, -3)
        y = self.conv(x_transpose)
        y = self.softmax(y)
        y = y.transpose(-2, -3)
        return y * self.C * x


class LMDA_Backbone(nn.Module):
    def __init__(self, chans, samples, depth=9, kernel=75, channel_depth1=24,
                 channel_depth2=9, avepool=5,ave_depth=1):
        super(LMDA_Backbone, self).__init__()
        self.channel_weight = nn.Parameter(torch.randn(depth, 1, chans), requires_grad=True)
        nn.init.xavier_uniform_(self.channel_weight.data)

        self.time_conv = nn.Sequential(
            nn.Conv2d(depth, channel_depth1, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(channel_depth1),
            nn.Conv2d(channel_depth1, channel_depth1, kernel_size=(1, kernel),
                      groups=channel_depth1, bias=False),
            nn.BatchNorm2d(channel_depth1),
            nn.GELU(),
        )

        self.chanel_conv = nn.Sequential(
            nn.Conv2d(channel_depth1, channel_depth2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(channel_depth2),
            nn.Conv2d(channel_depth2, channel_depth2, kernel_size=(chans, 1),
                      groups=channel_depth2, bias=False),
            nn.BatchNorm2d(channel_depth2),
            nn.GELU(),
        )

        self.norm = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 1, avepool)),
            nn.Dropout(p=0.65),
        )

        # 用一次前向确定输出维度
        with torch.no_grad():
            dummy = torch.ones((1, 1, chans, samples))
            out = torch.einsum('bdcw, hdc->bhcw', dummy, self.channel_weight)
            out = self.time_conv(out)
            self.depthAttention = EEGDepthAttention(out.shape[-1], out.shape[1], k=7)
            out = self.depthAttention(out)
            out = self.chanel_conv(out)
            out = self.norm(out)
            self.flatten_dim = out.numel()

    def forward(self, x):
        x = torch.einsum('bdcw, hdc->bhcw', x, self.channel_weight)
        x = self.time_conv(x)
        x = self.depthAttention(x)
        x = self.chanel_conv(x)
        x = self.norm(x)
        x = torch.flatten(x, 1)
        return x


class LMDA_Multi_Task(nn.Module):
    def __init__(self, chans=22, samples=1125, num_classes1=2,
                 num_classes2=4, depth=9, kernel=75,
                 channel_depth1=24, channel_depth2=9, avepool=5,ave_depth=1):
        super(LMDA_Multi_Task, self).__init__()
        self.backbone = LMDA_Backbone(
            chans=chans, samples=samples, depth=depth,
            kernel=kernel, channel_depth1=channel_depth1,
            channel_depth2=channel_depth2, avepool=avepool,ave_depth = ave_depth,
        )
        self.classifier1 = nn.Linear(self.backbone.flatten_dim, num_classes1)
        self.classifier2 = nn.Linear(self.backbone.flatten_dim, num_classes2)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, return_feature=False):
        x = self.backbone(x)
        if return_feature:
            return x
        return self.classifier1(x), self.classifier2(x)


if __name__ == '__main__':
    model = LMDA_Multi_Task(num_classes1=2, num_classes2=4,
                            chans=22, samples=1000,
                            channel_depth1=24, channel_depth2=9,
                            avepool=100).cuda()
    x = torch.randn(64, 1, 22, 1000).cuda()
    out1, out2 = model(x)
    summary(model, input_size=(1, 22, 1000), batch_size=64)
    print("out1 shape:", out1.shape)
    print("out2 shape:", out2.shape)
