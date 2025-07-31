import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary
import mne
from scipy.spatial.distance import cdist



###########################
# 1) Chebyshev相关辅助函数
###########################
def normalize_A(A, lmax=2):
    """
    把可学习的邻接矩阵 A 做以下处理:
      1) ReLU 保证非负
      2) 把对角置 0 (去自连接) => A = A * (ones - I)
      3) 对称化 A = A + A^T
      4) 构建拉普拉斯 L = I - D^{-1/2} A D^{-1/2}
      5) 归一到 [-1, 1]: Lnorm = (2 * L / lmax) - I
    注意：为避免张量在不同设备上冲突，这里根据 A 的 device 动态生成 I, ones。
    """
    device = A.device
    A = F.relu(A)
    N = A.shape[0]
    I = torch.eye(N, device=device)
    ones_ = torch.ones(N, N, device=device)

    # 去除自连接 & 对称化
    A = A * (ones_ - I)
    A = A + A.t()

    # 构造 L = I - D^{-1/2} A D^{-1/2}
    d = torch.sum(A, dim=1)               # (N,)
    d_inv_sqrt = 1.0 / torch.sqrt(d + 1e-10)
    D_inv_sqrt = torch.diag_embed(d_inv_sqrt)  # (N, N)
    L = I - (D_inv_sqrt @ A @ D_inv_sqrt)

    # Chebyshev常见做法：把 L 映射到 [-1, 1]
    Lnorm = (2.0 * L / lmax) - I
    return Lnorm

def generate_cheby_adj(L, K):
    """
    生成 K 阶 Chebyshev 多项式列表: [T_0(L), T_1(L), ..., T_{K-1}(L)]
      T_0(L)=I, T_1(L)=L,
      T_k(L)=2L * T_{k-1}(L) - T_{k-2}(L)
    """
    device = L.device
    N = L.shape[0]

    support = []
    for i in range(K):
        if i == 0:
            support.append(torch.eye(N, device=device))
        elif i == 1:
            support.append(L)
        else:
            temp = 2.0 * (L @ support[-1]) - support[-2]
            support.append(temp)
    return support


###########################
# 2) GraphConvolution
###########################
class GraphConvolution(nn.Module):
    """
    针对输入形状 (N, in_channels, C) 的单层 GCN:
      - (N, in_channels, C) 先 permute => (N, C, in_channels)
      - 每个样本的通道维 (C) 与邻接矩阵 (C, C) 相乘
      - 再与可学习权重 weight (in_channels, out_channels) 相乘
      - 输出 (N, out_channels, C)

    其中:
      N = B * W (合并批次和时间等维度)
      C = EEG通道数 (节点数)
    """
    def __init__(self, in_channels, out_channels, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # weight: (in_channels, out_channels)
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        nn.init.xavier_normal_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

    def forward(self, x, A):
        """
        x: (N, in_channels, C)
        A: (C, C)
        return: (N, out_channels, C)
        """
        # 1) Permute => (N, C, in_channels)
        N, F_in, C = x.shape
        x_perm = x.permute(0, 2, 1).contiguous()  # => (N, C, F_in)

        # 2) 批量邻接乘法: (N, C, C) x (N, C, F_in) => (N, C, F_in)
        A_expanded = A.unsqueeze(0).expand(N, -1, -1)  # => (N, C, C)
        out = torch.bmm(A_expanded, x_perm)            # => (N, C, F_in)

        # 3) Linear transform: (F_in -> F_out)
        out = out.view(N*C, F_in)                      # => (N*C, F_in)
        out = out @ self.weight                        # => (N*C, out_channels)
        out = out.view(N, C, self.out_channels)        # => (N, C, out_channels)

        # 4) 加上 bias 并 permute 回 (N, out_channels, C)
        if self.bias is not None:
            out = out + self.bias  # (广播到 (N, C, out_channels))

        out = out.permute(0, 2, 1).contiguous()        # => (N, out_channels, C)
        return out


###########################
# 3) Chebynet
###########################
class Chebynet(nn.Module):
    """
    K 阶 Chebyshev 多项式展开:
      输入 x: (N, in_channels, C)
      输出: (N, out_channels, C)
    """
    def __init__(self, in_channels, K, out_channels):
        super(Chebynet, self).__init__()
        self.K = K
        self.gc_list = nn.ModuleList()
        for _ in range(K):
            self.gc_list.append(GraphConvolution(in_channels, out_channels, bias=True))

    def forward(self, x, L):
        """
        x: (N, in_channels, C)
        L: (C, C)  # 归一化后的拉普拉斯
        return:
          - result: (N, out_channels, C)
        """
        # 原先还返回 adj_list，但为了兼容 torchsummary，这里只返回 result
        # 若想调试或可视化 adj_list，可自行添加
        adj_list = generate_cheby_adj(L, self.K)

        # 累加每一阶 T_k(L) 的卷积结果
        for i, gc in enumerate(self.gc_list):
            if i == 0:
                result = gc(x, adj_list[i])  # => (N, out_channels, C)
            else:
                result += gc(x, adj_list[i])

        result = F.relu(result)
        return result  # 只返回主输出，避免 torchsummary 冲突


###########################
# 4) EEGDepthAttention & MLP
###########################
class EEGDepthAttention(nn.Module):
    def __init__(self, W, C, k=7):
        super(EEGDepthAttention, self).__init__()
        self.C = C
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, W))
        self.conv = nn.Conv2d(1, 1, kernel_size=(k, 1), padding=(k // 2, 0), bias=True)
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, x):
        """
        x: (B, channel_depth1, chans, W)
        """
        x_pool = self.adaptive_pool(x)          # => (B, channel_depth1, 1, W)
        x_transpose = x_pool.transpose(-2, -3)  # => (B, channel_depth1, W, 1)
        y = self.conv(x_transpose)              # => (B, channel_depth1, W, 1)
        y = self.softmax(y)                     # => 同形状
        y = y.transpose(-2, -3)                 # => (B, channel_depth1, 1, W)
        return y * self.C * x                   # => (B, channel_depth1, chans, W)


class MLP(nn.Module):
    def __init__(self, feature, hidden_size):
        super(MLP, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 4)
        )

    def forward(self, x):
        return self.classifier(x)


###########################
# 5) 最终 LMDA-Net
###########################
class LMDA(nn.Module):
    """
    LMDA-Net + Chebynet
      - 输入 x: (B, 1, chans, samples)
      - 导联权重 => (B, depth, chans, samples)
      - 时间卷积 => (B, channel_depth1, chans, W')
      - DepthAttention
      - chanel_conv_part1 => (B, channel_depth2, chans, W')
      - 归一化邻接 + Chebynet => 先把 (B, channel_depth2, chans, W') reshape => (B*W', channel_depth2, chans)，做图卷积 => (B*W', channel_depth2, chans)，再 reshape 回来
      - chanel_conv_part2 => (B, channel_depth2, 1, W'')
      - norm => flatten => MLP
    """
    def __init__(self,
                 prune_lambda=1e-4,    # 稀疏正则化项的超参数
                 prune_ratio=0.9,      # 剪枝保留比例
                 classification=True,
                 chans=22,
                 samples=1125,
                 num_classes=4,
                 depth=9,
                 kernel=75,
                 channel_depth1=24,
                 channel_depth2=9,
                 ave_depth=1,
                 avepool=5,
                 adj_init=None,
                 cheby_k=3):
        super(LMDA, self).__init__()
        self.classification = classification
        self.ave_depth = ave_depth
        self.prune_ratio = prune_ratio
        # (1) 导联权重
        self.channel_weight = nn.Parameter(torch.randn(depth, 1, chans), requires_grad=True)
        nn.init.xavier_uniform_(self.channel_weight.data)

        # (2) 时间卷积
        self.time_conv = nn.Sequential(
            nn.Conv2d(depth, channel_depth1, kernel_size=(1, 1), groups=1, bias=False),
            nn.BatchNorm2d(channel_depth1),
            nn.Conv2d(channel_depth1, channel_depth1, kernel_size=(1, kernel), groups=channel_depth1, bias=False),
            nn.BatchNorm2d(channel_depth1),
            nn.GELU(),
        )

        # (3) 通道卷积拆分 part1
        self.chanel_conv_part1 = nn.Sequential(
            nn.Conv2d(channel_depth1, channel_depth2, kernel_size=(1, 1), groups=1, bias=False),
            nn.BatchNorm2d(channel_depth2),
        )

        # (3.1) 可学习邻接矩阵 A
        if adj_init is not None:
            self.A = nn.Parameter(torch.tensor(adj_init), requires_grad=True)
        else:
            self.A = nn.Parameter(torch.empty(chans, chans), requires_grad=True)
            nn.init.uniform_(self.A, 0.01, 0.5)
        self.A_mask = nn.Parameter(torch.ones_like(self.A), requires_grad=True)


        # (3.2) Chebynet
        # self.chebynet = Chebynet(
        #     in_channels=channel_depth2,  # 输入特征维度
        #     K=cheby_k,
        #     out_channels=channel_depth2  # 输出特征维度
        # )
        self.cheby_layers = nn.ModuleList([
            Chebynet(in_channels=channel_depth2, K=cheby_k, out_channels=channel_depth2)
            for _ in range(5)
        ])
        # (3.1) 可学习邻接矩阵 A
        if adj_init is not None:
            self.A = nn.Parameter(torch.tensor(adj_init), requires_grad=True)
        else:
            self.A = nn.Parameter(torch.empty(chans, chans), requires_grad=True)
            nn.init.uniform_(self.A, 0.01, 0.5)
        self.A_mask = nn.Parameter(torch.ones_like(self.A), requires_grad=True)
        self.A_pruned = self.A * torch.sigmoid(self.A_mask)

        # (3.3) 通道卷积 part2 (深度可分卷积)
        self.chanel_conv_part2 = nn.Sequential(
            nn.Conv2d(channel_depth2, channel_depth2, kernel_size=(chans, 1), groups=channel_depth2, bias=False),
            nn.BatchNorm2d(channel_depth2),
            nn.GELU(),
        )

        # (4) 池化 + dropout
        self.norm = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 1, avepool)),
            nn.Dropout(p=0.65),
        )

        # (5) 预推断，确定最后 flatten 的特征数
        with torch.no_grad():
            tmp = torch.ones((1, 1, chans, samples))
            tmp = torch.einsum('bdcw, hdc->bhcw', tmp, self.channel_weight)  # => (1, depth, chans, samples)
            tmp = self.time_conv(tmp)  # => (1, channel_depth1, chans, W')
            tmp = self.chanel_conv_part1(tmp)  # => (1, channel_depth2, chans, W')

            A_tmp = self.A.to(tmp.device)
            L_norm = normalize_A(A_tmp)

            # (1) reshape => (B*W', channel_depth2, chans)
            B, F_in, C, W_ = tmp.shape
            tmp2 = tmp.permute(0, 3, 1, 2).contiguous()  # => (1, W', channel_depth2, chans)
            tmp2 = tmp2.view(B*W_, F_in, C)              # => (B*W', channel_depth2, chans)
            # tmp2 = self.chebynet(tmp2, L_norm)           # => 只返回 result
            for cheby in self.cheby_layers:
                tmp2 = cheby(tmp2, L_norm)

            # (2) reshape 回原形 => (B, channel_depth2, chans, W')
            tmp2 = tmp2.view(B, W_, F_in, C).permute(0, 2, 3, 1).contiguous()

            # (3) 通道卷积 part2 => => (1, channel_depth2, 1, newW?)
            tmp2 = self.chanel_conv_part2(tmp2)
            tmp2 = self.norm(tmp2)

        N, C_, H_, W_ = tmp2.shape
        final_features = C_ * H_ * W_

        # (6) 全连接 + MLP
        mlp_features = 5
        hidden_feature = 64
        self.fc1 = nn.Linear(final_features, mlp_features)
        self.bn1 = nn.BatchNorm1d(mlp_features)
        self.MLP = MLP(mlp_features, hidden_feature)
        self.softmax = nn.Softmax(dim=1)
        self.classifier = nn.Linear(final_features, num_classes)

        print('In LMDA-Net (Chebynet), final shape: ', (N, C_, H_, W_))
    def update_prune_mask(self):
        """
        更新剪枝比例：通过 Top-K 或阈值去除最小连接
        """
        threshold = torch.topk(torch.abs(self.A).view(-1), int(self.A.numel() * self.prune_ratio), largest=True)[0][-1]
        self.A_mask.data = (torch.abs(self.A) > threshold).float()
    def forward(self, x):
        """
        x shape: (B, 1, chans, samples)
        """
        device = x.device  # 只要 x 在 GPU 或 CPU 上，这里跟着即可

        # (1) 导联权重 => (B, depth, chans, samples)
        x = torch.einsum('bdcw, hdc->bhcw', x, self.channel_weight)

        # (2) 时间卷积 => (B, channel_depth1, chans, W')
        x_time = self.time_conv(x)

        # (2.1) DepthAttention
        depth_attention = EEGDepthAttention(
            W=x_time.shape[-1],
            C=x_time.shape[1],
            k=7
        ).to(device)
        x_time = depth_attention(x_time)

        # (3) 通道卷积 part1 => (B, channel_depth2, chans, W')
        x = self.chanel_conv_part1(x_time)

        # (3.1) Chebynet
        A_pruned = self.A * torch.sigmoid(self.A_mask)# 归一化邻接
        A_norm = normalize_A(A_pruned.to(device))
        # A_norm = normalize_A(self.A.to(device))  # 只学习原始图中已有的边不剪枝




        B, F_in, C, W_ = x.shape
        x_reshaped = x.permute(0, 3, 1, 2).contiguous()  # => (B, W', F_in, C)
        x_reshaped = x_reshaped.view(B*W_, F_in, C)      # => (B*W', channel_depth2, chans)

        # x_reshaped = self.chebynet(x_reshaped, A_norm)   # => (B*W', channel_depth2, chans)
        for cheby in self.cheby_layers:
            x_reshaped = cheby(x_reshaped, A_norm)
        # (3.2) reshape回 (B, channel_depth2, chans, W')
        x_reshaped = x_reshaped.view(B, W_, F_in, C)
        x_reshaped = x_reshaped.permute(0, 2, 3, 1).contiguous()

        # 通道卷积 part2 => (B, channel_depth2, 1, newW)
        x = self.chanel_conv_part2(x_reshaped)

        # (4) 池化 + Dropout
        x = self.norm(x)  # => (B, channel_depth2, 1, ?)

        # (5) Flatten + 全连接
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        # x = self.bn1(x)  # 如果需要 BN 打开
        if self.classification:
            x = self.MLP(x)        # => (B, 4)
            x = self.softmax(x)    # => (B, 4)
        return x

###########################
# 6) 运行测试
###########################
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LMDA(
        classification=True,
        chans=22,
        samples=1000,
        channel_depth1=24,
        channel_depth2=9,
        avepool=250 // 10,
        cheby_k=3
    ).to(device)

    # 生成一个测试输入
    a = torch.randn(64, 1, 22, 1000).to(device)  # (B, 1, chans, samples)
    out = model(a)  # => (64, num_classes=4)
    print("Output shape:", out.shape)

    model_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    # 显示模型结构
    summary(model, input_size=(1, 22, 1000), batch_size=64, device=str(device))
