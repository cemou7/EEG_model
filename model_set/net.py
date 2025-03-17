import torch
import torch.nn as nn
from torchsummary import summary


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=33, padding=16)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size=33, padding=16)
        self.bn2 = nn.BatchNorm1d(in_channels)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.dropout(out)
        return out


def square_activation(x):
    return torch.square(x)


def safe_log(x):
    return torch.clip(torch.log(x), min=1e-7, max=1e7)


class ShallowConvNet(nn.Module):
    def __init__(self, num_classes, chans, samples=1125):
        super(ShallowConvNet, self).__init__()
        self.conv_nums = 40
        self.features = nn.Sequential(
            nn.Conv2d(1, self.conv_nums, (1, 25)),
            nn.Conv2d(self.conv_nums, self.conv_nums, (chans, 1), bias=False),
            nn.BatchNorm2d(self.conv_nums)
        )
        self.avgpool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))
        self.dropout = nn.Dropout()
        out = torch.ones((1, 1, chans, samples))
        out = self.features(out)
        out = self.avgpool(out)
        n_out_time = out.cpu().data.numpy().shape
        self.classifier = nn.Linear(n_out_time[-1] * n_out_time[-2] * n_out_time[-3], num_classes)

    def forward(self, x):
        x = self.features(x)
        x = square_activation(x)
        x = self.avgpool(x)
        x = safe_log(x)
        x = self.dropout(x)
        features = torch.flatten(x, 1)  # 使用卷积网络代替全连接层进行分类, 因此需要返回x和卷积层个数
        cls = self.classifier(features)
        return cls


    
class EEGNet(nn.Module):
    def __init__(self, num_classes, chans=64, samples=128, dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):
        super(EEGNet, self).__init__()

        '''
        Inputs:
        
        nb_classes      : int, number of classes to classify
        Chans, Samples  : number of channels and time points in the EEG data
        dropoutRate     : dropout fraction
        kernLength      : length of temporal convolution in first layer. We found
                            that setting this to be half the sampling rate worked
                            well in practice. 
        F1, F2          : number of temporal filters (F1) and number of pointwise
                            filters (F2) to learn. Default: F1 = 8, F2 = F1 * D. 
        D               : number of spatial filters to learn within each temporal
                            convolution. Default: D = 2
        dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.
        '''

        # 根据dropout类型选择dropout层
        if dropoutType == 'SpatialDropout2D':
            self.dropoutType = nn.Dropout2d(dropoutRate)
        elif dropoutType == 'Dropout':
            self.dropoutType = nn.Dropout(dropoutRate)
        else:
            raise ValueError('dropoutType must be one of SpatialDropout2D or Dropout, passed as a string.')

        # 第一块：时间卷积 -> 批量归一化 -> 深度卷积 -> 批量归一化 -> 激活 -> 池化 -> dropout
        self.time_conv = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernLength), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, D*F1, (chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(D*F1),
            nn.ELU(inplace=True),
            nn.AvgPool2d((1, 4)),
            self.dropoutType
        )
        
        # 第二块：空间可分离卷积 -> 批量归一化 -> 激活 -> 池化 -> dropout
        self.separable_conv = nn.Sequential(
            nn.Conv2d(D*F1, D*F1, (1, 16), groups=D*F1, bias=False),
            nn.Conv2d(D*F1, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(inplace=True),
            nn.AvgPool2d((1, 8)),
            self.dropoutType
        )
        
        out = torch.ones((1, 1, chans, samples))
        out = self.time_conv(out)
        out = self.separable_conv(out)
        n_out_time = out.cpu().data.numpy().shape

        # 分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_out_time[-1] * n_out_time[-2] * n_out_time[-3], num_classes),
            nn.Softmax(dim=1)  # 使用dim=1，因为输入的是(batch_size, nb_classes)
        )

    def forward(self, x):
        x = self.time_conv(x)
        x = self.separable_conv(x)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    model = EEGNet(num_classes=4, chans=22, samples=1000, kernLength=512//2).cuda()
    a = torch.randn(64, 1, 22, 1000).cuda().float()
    l2 = model(a)
    model_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    summary(model, input_size=(1, 22, 1000), batch_size=32)
    # print(model)
    print(l2.shape)

