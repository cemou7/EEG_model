import torch
import torch.nn as nn
from torchsummary import summary


class ShallowConvNet(nn.Module):
    def __init__(self, num_classes, chans, samples):
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
        x = torch.square(x)
        x = self.avgpool(x)
        x = torch.clip(torch.log(x), min=1e-7, max=1e7)
        x = self.dropout(x)
        features = torch.flatten(x, 1)  # 使用卷积网络代替全连接层进行分类, 因此需要返回x和卷积层个数
        # cls = self.classifier(features)
        return features


if __name__ == '__main__':
    model = ShallowConvNet(num_classes=4, chans=22, samples=1125).cuda()
    a = torch.randn(64, 1, 22, 1125).cuda().float()
    l2 = model(a)
    model_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    summary(model, input_size=(1, 22, 1125), batch_size=32)
    print(l2.shape)

