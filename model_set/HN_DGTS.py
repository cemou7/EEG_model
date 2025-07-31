import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class DynamicGraphConvolution(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.linears = nn.ModuleList([
            nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])

    def forward(self, x):  # x: [B, C, D]
        B, C, D = x.shape
        A = self.compute_adjacency(x)  # [B, C, C]
        L = self.normalize_adjacency(A)  # Laplacian matrix

        out = x
        for layer in self.linears:
            out = F.relu(torch.bmm(L, layer(out)))
        return out  # [B, C, D]

    def compute_adjacency(self, x):
        # PCC-based adjacency matrix with self-attention weight
        x_centered = x - x.mean(dim=2, keepdim=True)
        cov = torch.bmm(x_centered, x_centered.transpose(1, 2))
        std = x.std(dim=2, keepdim=True)
        denom = torch.bmm(std, std.transpose(1, 2)) + 1e-8
        A = cov / denom  # Pearson Correlation Coefficient
        return A

    def normalize_adjacency(self, A):
        I = torch.eye(A.size(1)).to(A.device)
        A_hat = A + I  # add self-loop
        D = A_hat.sum(dim=2)
        D_inv_sqrt = torch.diag_embed(D.pow(-0.5))
        return torch.bmm(torch.bmm(D_inv_sqrt, A_hat), D_inv_sqrt)


class TemporalSelfAttention(nn.Module):
    def __init__(self, input_dim, heads=1):
        super().__init__()
        self.q_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, input_dim)
        self.v_proj = nn.Linear(input_dim, input_dim)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim)
        )
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

    def forward(self, x):  # x: [B, C, D]
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        out = self.norm1(context + x)
        out = self.norm2(out + self.ffn(out))
        return out


class CrossAttentionFusion(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.query_proj = nn.Linear(input_dim, input_dim)
        self.key_proj = nn.Linear(input_dim, input_dim)
        self.value_proj = nn.Linear(input_dim, input_dim)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim)
        )

    def forward(self, q, kv):
        Q = self.query_proj(q)
        K = self.key_proj(kv)
        V = self.value_proj(kv)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        return self.ffn(context + q)


class HN_DGTS(nn.Module):
    def __init__(self, channel_num, feature_dim, hidden_dim=64, num_classes=2):
        super().__init__()
        self.dgc = DynamicGraphConvolution(input_dim=feature_dim, hidden_dim=hidden_dim)
        self.tsar = TemporalSelfAttention(input_dim=feature_dim)
        self.caf1 = CrossAttentionFusion(input_dim=hidden_dim)
        self.caf2 = CrossAttentionFusion(input_dim=hidden_dim)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channel_num * hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):  # x: [B, C, D]
        spatial_feat = self.dgc(x)
        temporal_feat = self.tsar(x)

        # Cross-attention fusion
        fused1 = self.caf1(spatial_feat, temporal_feat)
        fused2 = self.caf2(temporal_feat, fused1)

        return self.classifier(fused2)
    

if __name__ == '__main__':
    

    model = HN_DGTS(channel_num=22, feature_dim=128, hidden_dim=64, num_classes=4).cuda()
    dummy_input = torch.randn(64, 22, 128).cuda().float()
    output = model(dummy_input)
    model_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    summary(model, input_size=(22, 128), batch_size=64)
    print(output.shape)