
__all__ = ['PatchTST']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append("/home/work3/wkh/CL-Model")

from collections import OrderedDict
from PatchTST_self_supervised.src.models.layers.pos_encoding import *
from PatchTST_self_supervised.src.models.layers.basics import *
from PatchTST_self_supervised.src.models.layers.attention import *


class ChannelFeatureFusion(nn.Module):
    def __init__(self, in_channels=22, out_channels=1):
        super(ChannelFeatureFusion, self).__init__()

        self.channel_Conv = nn.Sequential(
            # nn.Conv2d(1, 1, kernel_size=(1, 1), groups=1, bias=False),
            # nn.BatchNorm2d(1),
            nn.Conv2d(1, 1, kernel_size=(in_channels, 1), groups=out_channels, bias=False),
            nn.BatchNorm2d(1),
            nn.GELU(),
        )
        
    def forward(self, x):
        x = x.permute(0, 2, 1, 3).flatten(2, 3).unsqueeze(1)

        x = self.channel_Conv(x)

        x = x.reshape(64, 20, 1, 50)
        return x


# Cell
class PatchTST_self(nn.Module):
    """
    Output dimension: 
         [bs x target_dim x nvars] for prediction
         [bs x target_dim] for regression
         [bs x target_dim] for classification
         [bs x num_patch x n_vars x patch_len] for pretrain
    """
    def __init__(self, c_in:int, target_dim:int, patch_len:int, stride:int, num_patch:int, 
                 n_layers:int=3, d_model=128, n_heads=16, shared_embedding=True, d_ff:int=256, 
                 norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", 
                 res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, head_dropout = 0, 
                 head_type = "prediction", individual = False, 
                 y_range:Optional[tuple]=None, verbose:bool=False, **kwargs):

        super().__init__()

        # 添加通道卷积
        self.channel_conv = ChannelFeatureFusion()

        assert head_type in ['pretrain', 'prediction', 'regression', 'classification'], 'head type should be either pretrain, prediction, or regression'
        # Backbone
        self.backbone = PatchTSTEncoder(c_in, num_patch=num_patch, patch_len=patch_len, 
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, 
                                shared_embedding=shared_embedding, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, 
                                res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)

        # Head
        self.n_vars = c_in      # 7
        self.head_type = head_type

        if head_type == "pretrain":     # 128, 12, 0.2
            self.head = PretrainHead(d_model, patch_len, head_dropout) # custom head passed as a partial func with all its kwargs
        elif head_type == "prediction":
            self.head = PredictionHead(individual, self.n_vars, d_model, num_patch, target_dim, head_dropout)
        elif head_type == "regression":
            self.head = RegressionHead(self.n_vars, d_model, target_dim, head_dropout, y_range)
        elif head_type == "classification":
            self.head = ClassificationHead(self.n_vars, d_model, target_dim, head_dropout)
            # self.head = FlattenHead(self.n_vars, d_model, num_patch, target_dim, head_dropout)


    def forward(self, z):                             
        """
        z: tensor [bs x num_patch x n_vars x patch_len]   
        """   
        # z = self.channel_conv(z)
        
        z = self.backbone(z)                      # z: [bs x nvars x d_model x num_patch]       [64, 22, 128, 20]
        z = self.head(z)

        # z: [bs x target_dim x nvars] for prediction       [64, 96, 7]
        #    [bs x target_dim] for regression
        #    [bs x target_dim] for classification           [64, 4]
        #    [bs x num_patch x n_vars x patch_len] for pretrain
        return z


class RegressionHead(nn.Module):
    def __init__(self, n_vars, d_model, output_dim, head_dropout, y_range=None):
        super().__init__()
        self.y_range = y_range
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars*d_model, output_dim)

    def forward(self, x):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x output_dim]
        """
        x = x[:,:,:,-1]             # only consider the last item in the sequence, x: bs x nvars x d_model
        x = self.flatten(x)         # x: bs x nvars * d_model
        x = self.dropout(x)
        y = self.linear(x)         # y: bs x output_dim
        if self.y_range: y = SigmoidRange(*self.y_range)(y)        
        return y


class ClassificationHead(nn.Module):
    def __init__(self, n_vars, d_model, n_classes, head_dropout):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars*d_model, n_classes)

    def forward(self, x):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x n_classes]
        """
        x = x[:,:,:,-1]             # only consider the last item in the sequence, x: bs x nvars x d_model
        embed_x = self.flatten(x)         # x: bs x nvars * d_model
        x = self.dropout(embed_x)
        y = self.linear(x)         # y: bs x n_classes

        return embed_x, y


class FlattenHead(nn.Module):
    def __init__(self, n_vars, d_model, num_patch, n_classes, head_dropout):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(head_dropout)
        self.projection = nn.Linear(n_vars*d_model*num_patch, n_classes)

    def forward(self, x):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x n_classes]
        """
        x = self.flatten(x)         # x: bs x nvars * d_model
        x = self.dropout(x)
        embed_x = x.reshape(x.shape[0], -1)
        y = self.projection(embed_x)         # y: bs x n_classes

        return embed_x, y


class PredictionHead(nn.Module):
    def __init__(self, individual, n_vars, d_model, num_patch, forecast_len, head_dropout=0, flatten=False):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars
        self.flatten = flatten
        head_dim = d_model*num_patch

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(head_dim, forecast_len))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(head_dim, forecast_len)
            self.dropout = nn.Dropout(head_dropout)


    def forward(self, x):                     
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x forecast_len x nvars]
        """
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * num_patch]
                z = self.linears[i](z)                    # z: [bs x forecast_len]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)         # x: [bs x nvars x forecast_len]
        else:
            x = self.flatten(x)     # x: [bs x nvars x (d_model * num_patch)]    
            x = self.dropout(x)
            x = self.linear(x)      # x: [bs x nvars x forecast_len]
        return x.transpose(2,1)     # [bs x forecast_len x nvars]


class PretrainHead(nn.Module):
    def __init__(self, d_model, patch_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, patch_len)

    def forward(self, x):
        """
        x: tensor [bs x nvars x d_model x num_patch]    [64, 7, 128, 42]
        output: tensor [bs x num_patch x nvars x patch_len]     [64, 42, 7, 12]     原始信号
        """

        x = x.transpose(2,3)                     # [bs x nvars x num_patch x d_model]
        x = self.linear( self.dropout(x) )      # [bs x nvars x num_patch x patch_len]
        x = x.permute(0,2,1,3)                  # [bs x num_patch x nvars x patch_len]
        return x


class PatchTSTEncoder(nn.Module):
    def __init__(self, c_in, num_patch, patch_len, 
                 n_layers=3, d_model=128, n_heads=16, shared_embedding=True,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):

        super().__init__()
        self.n_vars = c_in
        self.num_patch = num_patch
        self.patch_len = patch_len
        self.d_model = d_model
        self.shared_embedding = shared_embedding        

        # Input encoding: projection of feature vectors onto a d-dim vector space
        if not shared_embedding: 
            self.W_P = nn.ModuleList()
            for _ in range(self.n_vars): self.W_P.append(nn.Linear(patch_len, d_model))
        else:
            self.W_P = nn.Linear(patch_len, d_model)      

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, num_patch, d_model)

        # Residual dropout 0.2
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(d_model, n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, 
                                    store_attn=store_attn)

    def forward(self, x) -> Tensor:          
        """
        x: tensor [bs x num_patch x nvars x patch_len]
        """
        bs, num_patch, n_vars, patch_len = x.shape
        # Input encoding
        if not self.shared_embedding:
            x_out = []
            for i in range(n_vars): 
                z = self.W_P[i](x[:,:,i,:])
                x_out.append(z)
            x = torch.stack(x_out, dim=2)
        else:
            x = self.W_P(x)                                                      # x: [bs x num_patch x nvars x d_model]        patch: 50 -> 128
        x = x.transpose(1,2)                                                     # x: [bs x nvars x num_patch x d_model]        [64, 22, 20, 128]

        u = torch.reshape(x, (bs*n_vars, num_patch, self.d_model) )              # u: [bs * nvars x num_patch x d_model]        [1408, 20, 128]
        u = self.dropout(u + self.W_pos)                                         # u: [bs * nvars x num_patch x d_model]

        # Transformer Encoder
        z = self.encoder(u)                                                      # z: [bs * nvars x num_patch x d_model]        
        z = torch.reshape(z, (-1,n_vars, num_patch, self.d_model))               # z: [bs x nvars x num_patch x d_model]
        z = z.permute(0,1,3,2)                                                   # z: [bs x nvars x d_model x num_patch]

        return z
    
    
# Cell
class TSTEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=3, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(d_model, n_heads=n_heads, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src:Tensor):
        """
        src: tensor [bs x q_len x d_model]
        """
        output = src
        scores = None
        if self.res_attention:          # 是否使用残差注意力（Residual Attention）
            for mod in self.layers: output, scores = mod(output, prev=scores)
            return output
        else:
            for mod in self.layers: output = mod(output)
            return output



class TSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, 
                activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads
        d_v = d_model // n_heads

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn


    def forward(self, src:Tensor, prev:Optional[Tensor]=None):
        """
        src: tensor [bs x q_len x d_model]
        """
        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev)
        else:
            src2, attn = self.self_attn(src, src, src)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src


import argparse
import os
def set_parser():
    parser = argparse.ArgumentParser()
    # Dataset and dataloader
    parser.add_argument('--dset_pretrain', type=str, default='etth1', help='dataset name')

    parser.add_argument('--context_points', type=int, default=1000, help='sequence length')         # 1000  1921
    parser.add_argument('--target_points', type=int, default=4, help='forecast horizon')            # class

    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for DataLoader')
    parser.add_argument('--scaler', type=str, default='standard', help='scale the input data')
    parser.add_argument('--features', type=str, default='M', help='for multivariate model or univariate model')
    # Patch
    parser.add_argument('--patch_len', type=int, default=50, help='patch length')                   # all  20/42  patch
    parser.add_argument('--stride', type=int, default=50, help='stride between patch')
    # RevIN
    parser.add_argument('--revin', type=int, default=1, help='reversible instance normalization')
    # Model args
    parser.add_argument('--n_layers', type=int, default=3, help='number of Transformer layers')
    parser.add_argument('--n_heads', type=int, default=16, help='number of Transformer heads')
    parser.add_argument('--d_model', type=int, default=128, help='Transformer d_model')
    parser.add_argument('--d_ff', type=int, default=512, help='Tranformer MLP dimension')    # MLP dimension = 4 * d_model
    parser.add_argument('--dropout', type=float, default=0.2, help='Transformer dropout')
    parser.add_argument('--head_dropout', type=float, default=0.2, help='head dropout')
    # Pretrain mask
    parser.add_argument('--mask_ratio', type=float, default=0.5, help='masking ratio for the input')        # mask 50 %
    # Optimization args
    parser.add_argument('--n_epochs_pretrain', type=int, default=50, help='number of pre-training epochs')  # epoch
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    # model id to keep track of the number of models saved
    parser.add_argument('--pretrained_model_id', type=int, default=1, help='id of the saved pretrained model')
    parser.add_argument('--model_type', type=str, default='based_model',
                        help='for multivariate model or univariate model')

    args = parser.parse_args()
    print('args:', args)
    # args.save_pretrained_model = 'patchtst_pretrained_cw'+str(args.context_points)+'_patch'+str(args.patch_len) + '_stride'+str(args.stride) + '_epochs-pretrain' + str(args.n_epochs_pretrain) + '_mask' + str(args.mask_ratio)  + '_model' + str(args.pretrained_model_id)
    # args.save_path = 'saved_models/' + args.dset_pretrain + '/masked_patchtst/' + args.model_type + '/'
    # if not os.path.exists(args.save_path): os.makedirs(args.save_path)
    return args


def get_model_self(c_in, head_type):
    """
    c_in: number of variables
    """
    args = set_parser()

    # get number of patches
    num_patch = (max(args.context_points, args.patch_len) - args.patch_len) // args.stride + 1  # 42
    print('number of patches:', num_patch)

    # get model
    model = PatchTST_self(c_in=c_in,
                        target_dim=args.target_points,
                        patch_len=args.patch_len,
                        stride=args.stride,
                        num_patch=num_patch,
                        n_layers=args.n_layers,
                        n_heads=args.n_heads,
                        d_model=args.d_model,
                        shared_embedding=True,
                        d_ff=args.d_ff,
                        dropout=args.dropout,
                        head_dropout=args.head_dropout,
                        act='relu',
                        head_type=head_type,         # 'pretrain'
                        res_attention=False
                        )
    # print out the model size
    print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))

    # model.load_state_dict(torch.load('PatchTST_self_supervised/saved_models/etth1/masked_patchtst/based_model/patchtst_pretrained_cw1000_patch50_stride50_epochs-pretrain50_mask0.5_model1.pth'))
    return model


if __name__ == '__main__':
    model = get_model_self(c_in=22, head_type='classification').cuda()

    state_dict = torch.load('PatchTST_self_supervised/saved_models/etth1/masked_patchtst/based_model/patchtst_pretrained_cw1000_patch50_stride50_epochs-pretrain100_mask0.5_model1.pth')
    # state_dict = torch.load('PatchTST_self_supervised/saved_models/etth1/masked_patchtst/based_model/patchtst_pretrained_cw1921_patch50_stride50_epochs-pretrain100_mask0.5_model1_context1921.pth')

    # 筛选并调整 'backbone.' 参数的键名
    backbone_state_dict = {k[len("backbone."):]: v for k, v in state_dict.items() if k.startswith("backbone.")}
    model.backbone.load_state_dict(backbone_state_dict)

    x = torch.randn(64, 20, 22, 50).cuda().float()
    _, outputs = model(x)

    # summary(self.model, input_size=(1000, 22), batch_size=8)
    print(model)
    print(outputs.shape)


