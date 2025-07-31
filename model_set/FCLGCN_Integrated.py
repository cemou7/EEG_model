import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

def generate_cheby_adj(L, K):
    support = []
    for i in range(K):
        if i == 0:
            support.append(torch.eye(L.shape[-1]).cuda())
        elif i == 1:
            support.append(L)
        else:
            temp = torch.matmul(2*L,support[-1],)-support[-2]
            support.append(temp)
    return support

class GraphConvolution(nn.Module):

    def __init__(self, in_channels, out_channels, bias=False):

        super(GraphConvolution, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels).cuda())
        nn.init.xavier_normal_(self.weight)
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels).cuda())
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        out = torch.matmul(adj, x)
        out = torch.matmul(out, self.weight)
        if self.bias is not None:
            return out + self.bias
        else:
            return out

class Chebynet(nn.Module):
    def __init__(self, in_channels, K, out_channels):
        super(Chebynet, self).__init__()
        self.K = K
        self.gc1 = nn.ModuleList()
        for i in range(K):
            self.gc1.append(GraphConvolution(in_channels, out_channels))

    def forward(self, x, L):
        adj = generate_cheby_adj(L, self.K)

        for i in range(len(self.gc1)):
            if i == 0:
                result = self.gc1[i](x, adj[i])
            else:
                result += self.gc1[i](x, adj[i])
        result = F.relu(result)
        return result, adj

def normalize_A(A, lmax=2):
    A=F.relu(A)
    N=A.shape[0]
    A=A*(torch.ones(N,N).cuda()-torch.eye(N,N).cuda())
    A=A+A.T
    d = torch.sum(A, 1)
    d = 1 / torch.sqrt((d + 1e-10))
    D = torch.diag_embed(d)
    L = torch.eye(N,N).cuda()-torch.matmul(torch.matmul(D, A), D)
    Lnorm=(2*L/lmax)-torch.eye(N,N).cuda()
    return Lnorm

class Linear(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        nn.init.xavier_normal_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, inputs):
        return self.linear(inputs)
class DGCNN(nn.Module):
    def __init__(self, in_channels, num_electrodes, k_adj, out_channels, num_classes=3):
        # in_channels(int): The feature dimension of each electrode.
        # num_electrodes(int): The number of electrodes.
        # k_adj(int): The number of graph convolutional layers.
        # out_channel(int): The feature dimension of  the graph after GCN.
        # num_classes(int): The number of classes to predict.
        super(DGCNN, self).__init__()
        self.K = k_adj
        self.layer1 = Chebynet(in_channels, k_adj, out_channels)
        # self.layer1 = GCN(in_channels, out_channels)

        self.BN1 = nn.BatchNorm1d(in_channels)
        self.fc = Linear(num_electrodes * out_channels, num_classes)
        self.A = nn.Parameter(torch.FloatTensor(num_electrodes, num_electrodes).cuda())
        nn.init.uniform_(self.A, 0.01, 0.5)

    def forward(self, x):
        x = self.BN1(x.transpose(1, 2)).transpose(1, 2)  # data can also be standardized offline
        L = normalize_A(self.A)
        result, adj = self.layer1(x, L)
        adj = adj[-1]
        # result = result.reshape(x.shape[0], -1)
        # result = self.fc(result)
        return result, adj


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """
        :param q: Queries, with dimension [B, L_q, D_q]
        :param k: Keys, with dimension [B, L_k, D_k]
        :param v: Values, with dimension [B, L_v, D_v], generally is k
        :param scale: Scaling factor, a float scalar
        :param attn_mask: Masking, with dimension [B, L_q, L_k]
        :return context: Context
        :return attention: Attention
        """
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if attn_mask:
            attention = attention.masked_fill_(attn_mask, -np.inf)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.bmm(attention, v)
        return context, attention


class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim, num_heads, attn_drop=0.0, proj_drop=0.0):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads, bias=False)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads, bias=False)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads, bias=False)

        self.dot_product_attention = ScaledDotProductAttention(attn_drop)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(proj_drop)
        # layer norm after multi-head attention
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query, attn_mask=None):
        # residual connection
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)
        # split by heads
        key = key.view(batch_size, -1, num_heads, dim_per_head).transpose(1, 2).reshape(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size, -1, num_heads, dim_per_head).transpose(1, 2).reshape(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size, -1, num_heads, dim_per_head).transpose(1, 2).reshape(batch_size * num_heads, -1, dim_per_head)

        if attn_mask:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)
        # scaled dot product attention
        scale = dim_per_head ** -0.5
        context, attention = self.dot_product_attention(
            query, key, value, scale, attn_mask)

        # concat heads
        context = context.view(batch_size, num_heads, -1, dim_per_head).transpose(1, 2) \
            .reshape(batch_size, -1, num_heads * dim_per_head)

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)


        return output, attention


class FCLGCN_Integrated(nn.Module):
    def __init__(self, K, T, num_channels, num_features, num_classes=3):
        super(FCLGCN_Integrated, self).__init__()
        self.K = K
        self.T = T
        self.num_channels = num_channels
        self.num_features = num_features

        # 动态图卷积模块（每个时间步一套卷积）
        self.convs = nn.ModuleList([
            DGCNN(num_features, num_channels, K, num_features) for _ in range(T)
        ])
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(num_features) for _ in range(T)
        ])

        # 频带注意力（多头自注意力）
        self.attention = MultiHeadAttention(model_dim=num_channels * num_features, num_heads=4)

        # GRU 时间建模模块
        self.gru = nn.GRU(input_size=num_channels * num_features, hidden_size=num_channels, batch_first=True)

        # 输出层：分类 + 对比学习表征
        self.linear = nn.Linear(num_channels * T, 128)
        self.projection = nn.Linear(128, num_classes)
        self.classifier = nn.Linear(num_channels * T, num_classes)

        # 初始化参数
        for param in self.parameters():
            if param.dim() < 2:
                nn.init.xavier_uniform_(param.unsqueeze(0))
            else:
                nn.init.xavier_uniform_(param)

    def forward(self, x):
        """
        x: List of torch_geometric.data.Data 对象（每个 batch 是一个样本）
        每个样本由 T 个时间片段组成，每个片段构图成图
        """
        batch_size = len(x)
        y_list = []
        for i in range(self.T):
            xi = [data.get_example(i) for data in x]  # 每个样本第i帧
            batch_i = Batch.from_data_list(xi)
            input_i = batch_i.x.reshape(batch_size, self.num_channels, self.num_features)
            yi, _ = self.convs[i](input_i)
            yi = yi.reshape(-1, self.num_features)
            yi = self.batch_norms[i](yi)
            y_list.append(yi)

        y = torch.stack(y_list)  # [T, B*N, F]
        y = torch.sigmoid(y)
        yt = y.transpose(0, 1)  # [B, T, N*F]
        y_seq = yt.reshape(batch_size, self.T, -1)  # [B, T, N*F]

        # 频带注意力 + GRU
        y_att, _ = self.attention(y_seq, y_seq, y_seq)
        y_gru, _ = self.gru(y_att)
        y_gru_out = torch.sigmoid(y_gru)

        # 输出表征 + 分类
        out_flat = y_gru_out.reshape(batch_size, -1)
        proj = F.normalize(torch.sigmoid(self.linear(out_flat)), dim=-1)
        y_pred = torch.sigmoid(self.classifier(out_flat))

        return proj.unsqueeze(1), y_pred  # 可选再输出 projection head


# 示例用法
model = FCLGCN_Integrated(K=3, T=6, num_channels=22, num_features=5, num_classes=3)
print(model)
