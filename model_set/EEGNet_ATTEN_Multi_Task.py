import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import random
import pdb
from torchsummary import summary




class EEGNet_ATTEN_Backbone(nn.Module):
    def __init__(self, Chans,kernLength1,kernLength2,kernLength3,F1,D, DOR): # DOR adjustable for generation of sub-networks
        super(EEGNet_ATTEN_Backbone, self).__init__()
        Chans = Chans
        dropoutRate = DOR
        kernLength1 = kernLength1
        kernLength2 = kernLength2
        kernLength3 = kernLength3
        F1 = F1
        D = D
        F2 = F1 * D
        self.cbam1 =CBAM(channel= F2, reduction= 8)
        self.cbam2 =CBAM(channel= F2, reduction= 8)
        self.cbam3 =CBAM(channel= F2, reduction= 8)
        
        # MTSF
        self.features1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernLength1), padding=(0, kernLength1 // 2), bias=False),
            nn.BatchNorm2d(F1),
            
            nn.Conv2d(F1, F2, (Chans, 1), groups=D, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(inplace=True),
            nn.MaxPool2d((1,8)),
            nn.Dropout(dropoutRate),

            SeparableConv2d(F2, F2, kernel_size=(1, 16), padding=(0, 8)),
            nn.BatchNorm2d(F2),
            nn.ELU(inplace=True),
            nn.MaxPool2d((1, 8))
        )

        self.features2 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernLength2), padding=(0, kernLength2 // 2), bias=False),
            nn.BatchNorm2d(F1),
            
            nn.Conv2d(F1, F2, (Chans, 1), groups=D, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(inplace=True),
            nn.MaxPool2d((1,8)),
            nn.Dropout(dropoutRate),

            SeparableConv2d(F2, F2, kernel_size=(1,16), padding=(0, 8)),
            nn.BatchNorm2d(F2),
            nn.ELU(inplace=True),
            nn.MaxPool2d((1, 8))
        )
        self.features3 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernLength3), padding=(0, kernLength3 // 2), bias=False),
            nn.BatchNorm2d(F1),

            nn.Conv2d(F1, F2, (Chans, 1), groups=D, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(inplace=True),
            nn.MaxPool2d((1,8)),
            nn.Dropout(dropoutRate),

            SeparableConv2d(F2, F2, kernel_size=(1,16), padding=(0, 8)),
            nn.BatchNorm2d(F2),
            nn.ELU(inplace=True),
            nn.MaxPool2d((1, 8))
        )

        self.dropout = nn.Dropout(dropoutRate)
        self.inplanes = F2*3

    def forward(self, x,return_feature=False, return_partial=False):
        x = x.unsqueeze(1)
        x1 = self.features1(x)
        x1 = self.cbam1(x1)

        x2 = self.features2(x)
        x2 = self.cbam2(x2)
        if return_partial:  # 用于任务1：只取 x1 特征做模态分类
            x2_flat = torch.flatten(x2, start_dim=1)
            return x2_flat        
        x3 = self.features3(x)
        x3 = self.cbam3(x3)
        x_concat = torch.cat((x1, x2, x3), dim=1)
        x_concat = self.dropout(x_concat)
        # x_concat = self.ADR(x_concat)
        if return_feature:
            x_concat = torch.flatten(x_concat, start_dim=1)
            return x_concat
        x_concat = x_concat.squeeze(2)
        return x_concat

# 全backbone
class EEGNet_ATTEN_Multi_Task(nn.Module):
    def __init__(self, Chans,kernLength1,kernLength2,kernLength3,F1,D,num_classes1,num_classes2, DOR):
        super(EEGNet_ATTEN_Multi_Task, self).__init__()
        self.backbone = EEGNet_ATTEN_Backbone(Chans,kernLength1,kernLength2,kernLength3,F1,D, DOR)
        self.classifier1 = target_classifier(num_classes1)
        self.classifier2 = target_classifier(num_classes2)
    def forward(self, x,return_feature=False):
        x = self.backbone(x)
        if return_feature:
            return x
        return self.classifier1(x), self.classifier2(x)

# 半backbone
# class EEGNet_ATTEN_Multi_Task(nn.Module):
#     def __init__(self, Chans,kernLength1,kernLength2,kernLength3,F1,D,num_classes1,num_classes2, DOR):
#         super(EEGNet_ATTEN_Multi_Task, self).__init__()
#         self.backbone = EEGNet_ATTEN_Backbone(Chans,kernLength1,kernLength2,kernLength3,F1,D, DOR)
#         self.classifier1 = target_classifier(num_classes1)
#         self.classifier2 = target_classifier(num_classes2)
#     def forward(self, x,return_feature=False):
#         if return_feature:
#             x = self.backbone(x,return_feature=True)
#             return x
#         # task1 用中间特征 x1 做模态分类（静息 vs 想象）
#         feat_task1 = self.backbone(x, return_partial=True)
#         out1 = self.classifier1(feat_task1)
#         # task2 用整体特征做运动想象 4 分类
#         feat_task2 = self.backbone(x)
#         out2 = self.classifier2(feat_task2)       
#         return out1, out2

class SeparableConv2d(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel_size: tuple, padding: tuple = 0):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.padding = padding
        self.depthwise_conv = nn.Conv2d(self.c_in, self.c_in, kernel_size=self.kernel_size,
                                        padding=self.padding, groups=self.c_in)
        self.conv2d_1x1 = nn.Conv2d(self.c_in, self.c_out, kernel_size=1)

    def forward(self, x: torch.Tensor):
        y = self.depthwise_conv(x)
        y = self.conv2d_1x1(y)
        return y


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ELU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# Channel Attention Module
class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // reduction, num_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# Spatial Attention Module
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

# Combined CBAM Module
class CBAM(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(channel, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x) * x
        return x

class target_classifier(nn.Module):  # 更强大的分类头
    def __init__(self, num_classes_target):
        super(target_classifier, self).__init__()
        self.classifier = ResidualMLP(input_dim=1440, hidden_dim=512, num_classes=num_classes_target)

    def forward(self, x):
        x_flat = x.reshape(x.shape[0], -1)
        return self.classifier(x_flat)



class ResidualMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.out = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.elu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = x + residual
        x = self.out(x)
        return x


# class target_classifier(nn.Module):  # Classification head
#     def __init__(self, num_classes_target):
#         super(target_classifier, self).__init__()
#         self.logits = nn.Linear(1440, 64)
#         self.logits_simple = nn.Linear(64, num_classes_target)

#     def forward(self, emb):
#         """2-layer MLP"""
#         emb_flat = emb.reshape(emb.shape[0], -1)
#         emb = torch.sigmoid(self.logits(emb_flat))
#         pred = self.logits_simple(emb)
#         return pred


# class target_classifier(nn.Module):  # Classification head with deeper FC
#     def __init__(self, num_classes_target):
#         super(target_classifier, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(1440, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Dropout(0.3),

#             nn.Linear(512, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Dropout(0.3),

#             nn.Linear(128, 64)  # 相当于原来 self.logits
#         )
#         self.logits_simple = nn.Linear(64, num_classes_target)

#     def forward(self, emb):
#         emb_flat = emb.reshape(emb.shape[0], -1)
#         emb = torch.sigmoid(self.fc(emb_flat))  # 保留 sigmoid
#         pred = self.logits_simple(emb)
#         return pred

class Classifier(nn.Module):
    def __init__(self, num_classes,CNNoutput_channel,final_out_channels):
        super(Classifier, self).__init__()
        num_classes = num_classes
        self.FC1 = nn.Linear(CNNoutput_channel * final_out_channels, 1024)
        self.elu = nn.ELU(inplace=True)
        self.FC2 = nn.Linear(1024, num_classes)
        self.sf = nn.Softmax(dim=1)
    def forward(self, input):

        logits = self.FC1(input)
        logits = self.elu(logits)
        logits = self.FC2(logits)
        logits = self.sf(logits)
        return logits

if __name__ == '__main__':
    model = EEGNet_ATTEN_Multi_Task(Chans=22,kernLength1=36,kernLength2=24,kernLength3=18,F1=16,D=2,num_classes1=2,num_classes2=4,DOR=0.5).cuda()
    x = torch.randn(64, 22, 1000).cuda()
    out1,out2= model(x)
    summary(model, input_size=(22, 1000), batch_size=64)
    print("out1 shape:", out1.shape)
    print("out2 shape:", out2.shape)
