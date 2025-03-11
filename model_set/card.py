from tensorboard import summary
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
import argparse
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="3"          # if use '2,1', then in pytorch, gpu2 has id 0


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class Model(nn.Module):
    def __init__(self, config, **kwargs):
        
        super().__init__()    
        self.model = CARDformer(config)
        self.task_name = config.task_name
    def forward(self, x, *args, **kwargs):    

        x = x.permute(0,2,1)    

        mask = 0         
        x= self.model(x,mask = mask)
        if self.task_name != 'classification':
            x = x.permute(0,2,1)   
        return x
    
    
class CARDformer(nn.Module):
    def __init__(self, 
                 config,**kwargs):
        
        super().__init__()
        
        self.patch_len  = config.patch_len
        self.stride = config.stride
        self.d_model = config.d_model
        self.task_name = config.task_name
        patch_num = int((config.seq_len - self.patch_len)/self.stride + 1)
        self.patch_num = patch_num
        self.W_pos_embed = nn.Parameter(torch.randn(patch_num,config.d_model)*1e-2)
        self.model_token_number = 0
        
        if self.model_token_number > 0:
            self.model_token = nn.Parameter(torch.randn(config.enc_in,self.model_token_number,config.d_model)*1e-2)
        
        
        self.total_token_number = (self.patch_num  + self.model_token_number + 1)
        config.total_token_number = self.total_token_number
             
        self.W_input_projection = nn.Linear(self.patch_len, config.d_model)  
        self.input_dropout  = nn.Dropout(config.dropout) 
        
                
        self.use_statistic = config.use_statistic
        self.W_statistic = nn.Linear(2,config.d_model) 
        self.cls = nn.Parameter(torch.randn(1,config.d_model)*1e-2)
        
        
        if config.task_name == 'long_term_forecast' or config.task_name == 'short_term_forecast':
            self.W_out = nn.Linear((patch_num+1+self.model_token_number)*config.d_model, config.pred_len) 
        elif config.task_name == 'imputation' or config.task_name == 'anomaly_detection':
            self.W_out = nn.Linear((patch_num+1+self.model_token_number)*config.d_model, config.seq_len) 
        elif config.task_name == 'classification':
            self.W_out = nn.Linear(config.d_model*config.enc_in, config.num_class)

        
        self.Attentions_over_token = nn.ModuleList([Attenion(config) for i in range(config.e_layers)])
        self.Attentions_over_channel = nn.ModuleList([Attenion(config,over_hidden = True) for i in range(config.e_layers)])
        self.Attentions_mlp = nn.ModuleList([nn.Linear(config.d_model,config.d_model)  for i in range(config.e_layers)])
        self.Attentions_dropout = nn.ModuleList([nn.Dropout(config.dropout)  for i in range(config.e_layers)])
        self.Attentions_norm = nn.ModuleList([nn.Sequential(Transpose(1,2), nn.BatchNorm1d(config.d_model,momentum = config.momentum), Transpose(1,2)) for i in range(config.e_layers)])       
            

    def forward(self, z,*args, **kwargs):     

        b,c,s = z.shape

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast' or self.task_name == 'anomaly_detection':
            z_mean = torch.mean(z,dim = (-1),keepdims = True)
            z_std = torch.std(z,dim = (-1),keepdims = True)
            z =  (z - z_mean)/(z_std + 1e-4)
        
        elif self.task_name == 'imputation':     
            mask = kwargs['mask'].permute(0,2,1) 
            z_mean = torch.sum(z, dim=-1) / torch.sum(mask == 1, dim=-1)
            z_mean = z_mean.unsqueeze(-1)
            z = z - z_mean
            z = z.masked_fill(mask == 0, 0)
            z_std = torch.sqrt(torch.sum(z * z, dim=-1) /
                           torch.sum(mask == 1, dim=-1) + 1e-5)
            z_std = z_std.unsqueeze(-1)
            z /= z_std + 1e-4
            
        zcube = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)                 # [bs, channel, patch_num, patch_size]    [16, 22, 124, 16]    
        z_embed = self.input_dropout(self.W_input_projection(zcube))+ self.W_pos_embed        # [bs, channel, patch_num, d_model]       [16, 22, 124, 512]
        
        if self.use_statistic:
            z_stat = torch.cat((z_mean,z_std),dim = -1)
            if z_stat.shape[-2]>1:
                z_stat = (z_stat - torch.mean(z_stat,dim =-2,keepdims = True))/( torch.std(z_stat,dim =-2,keepdims = True)+1e-4)
            z_stat = self.W_statistic(z_stat)
            z_embed = torch.cat((z_stat.unsqueeze(-2),z_embed),dim = -2) 
        else:
            cls_token = self.cls.repeat(z_embed.shape[0],z_embed.shape[1],1,1)
            z_embed = torch.cat((cls_token,z_embed),dim = -2)                                 # add cls_token       [16, 22, 125, 512]

        inputs = z_embed
        b,c,t,h = inputs.shape 
        for a_2,a_1,mlp,drop,norm  in zip(self.Attentions_over_token, self.Attentions_over_channel,self.Attentions_mlp ,self.Attentions_dropout,self.Attentions_norm ):
            output_1 = a_1(inputs.permute(0,2,1,3)).permute(0,2,1,3)
            output_2 = a_2(output_1)
            outputs = drop(mlp(output_1+output_2))+inputs
            outputs = norm(outputs.reshape(b*c,t,-1)).reshape(b,c,t,-1) 
            inputs = outputs
        
        if self.task_name != 'classification':
            z_out = self.W_out(outputs.reshape(b,c,-1))  
            z = z_out *(z_std+1e-4)  + z_mean 
        else:
            z = self.W_out(torch.mean(outputs[:,:,:,:],dim = -2).reshape(b,-1))               # mean!!!  [16, 22, 125, 512] -> [16, 22, 512] -> [16, 11264] -> [16, 4]
        return z
    

class Attenion(nn.Module):
    def __init__(self,config, over_hidden = False,trianable_smooth = False,untoken = False, *args, **kwargs):
        super().__init__()
        
        self.over_hidden = over_hidden
        self.untoken = untoken
        self.n_heads = config.n_heads
        self.c_in = config.enc_in
        self.qkv = nn.Linear(config.d_model, config.d_model * 3, bias=True)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.head_dim = config.d_model // config.n_heads
        

        self.dropout_mlp = nn.Dropout(config.dropout)
        self.mlp = nn.Linear( config.d_model,  config.d_model)
        

        self.norm_post1  = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(config.d_model,momentum = config.momentum), Transpose(1,2))
        self.norm_post2  = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(config.d_model,momentum = config.momentum), Transpose(1,2))
        
        self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(config.d_model,momentum = config.momentum), Transpose(1,2))
        
        
        self.dp_rank = config.dp_rank
        self.dp_k = nn.Linear(self.head_dim, self.dp_rank)
        self.dp_v = nn.Linear(self.head_dim, self.dp_rank)
        
        self.ff_1 = nn.Sequential(nn.Linear(config.d_model, config.d_ff, bias=True),
                        nn.GELU(),
                        nn.Dropout(config.dropout),
                        nn.Linear(config.d_ff, config.d_model, bias=True)
                       )
        
        self.ff_2= nn.Sequential(nn.Linear(config.d_model, config.d_ff, bias=True),
                        nn.GELU(),
                        nn.Dropout(config.dropout),
                        nn.Linear(config.d_ff, config.d_model, bias=True)
                                )     
        self.merge_size = config.merge_size

        ema_size = max(config.enc_in,config.total_token_number,config.dp_rank)
        ema_matrix = torch.zeros((ema_size,ema_size))
        alpha = config.alpha
        ema_matrix[0][0] = 1
        for i in range(1,config.total_token_number):
            for j in range(i):
                ema_matrix[i][j] =  ema_matrix[i-1][j]*(1-alpha)
            ema_matrix[i][i] = alpha
        self.register_buffer('ema_matrix',ema_matrix)
       
    def ema(self,src):
        return torch.einsum('bnhad,ga ->bnhgd',src,self.ema_matrix[:src.shape[-2],:src.shape[-2]])
        
        
    def ema_trianable(self,src):
        alpha = F.sigmoid(self.alpha)
        
        weights = alpha * (1 - alpha) ** self.arange[-src.shape[-2]:]
 

        w_f = torch.fft.rfft(weights,n = src.shape[-2]*2)
        src_f = torch.fft.rfft(src.float(),dim = -2,n = src.shape[-2]*2)    
        src_f = (src_f.permute(0,1,2,4,3)*w_f)
        src1 =torch.fft.irfft(src_f.float(),dim = -1,n=src.shape[-2]*2)[...,:src.shape[-2]].permute(0,1,2,4,3)  #.half()
        return src1


    def dynamic_projection(self,src,mlp):
        src_dp = mlp(src)
        src_dp = F.softmax(src_dp,dim = -1)
        src_dp = torch.einsum('bnhef,bnhec -> bnhcf',src,src_dp)
        return src_dp
        
        
    def forward(self, src, *args,**kwargs):

        B,nvars, H, C, = src.shape
        
        
        qkv = self.qkv(src).reshape(B,nvars, H, 3, self.n_heads, C // self.n_heads).permute(3, 0, 1,4, 2, 5)
   

        q, k, v = qkv[0], qkv[1], qkv[2]
    
        if not self.over_hidden: 
        
            attn_score_along_token = torch.einsum('bnhed,bnhfd->bnhef', self.ema(q), self.ema(k))/ self.head_dim ** -0.5

            attn_along_token = self.attn_dropout(F.softmax(attn_score_along_token, dim=-1) )
   
            output_along_token = torch.einsum('bnhef,bnhfd->bnhed', attn_along_token, v)


            
        else:

            v_dp,k_dp = self.dynamic_projection(v,self.dp_v) , self.dynamic_projection(k,self.dp_k)
            attn_score_along_token = torch.einsum('bnhed,bnhfd->bnhef', self.ema(q), self.ema(k_dp))/ self.head_dim ** -0.5
            

            attn_along_token = self.attn_dropout(F.softmax(attn_score_along_token, dim=-1) )
            output_along_token = torch.einsum('bnhef,bnhfd->bnhed', attn_along_token, v_dp)

        
        
        attn_score_along_hidden = torch.einsum('bnhae,bnhaf->bnhef', q,k)/ q.shape[-2] ** -0.5
        attn_along_hidden = self.attn_dropout(F.softmax(attn_score_along_hidden, dim=-1) )    
        output_along_hidden = torch.einsum('bnhef,bnhaf->bnhae', attn_along_hidden, v)



        merge_size = self.merge_size
        if not self.untoken:
            output1 = rearrange(output_along_token.reshape(B*nvars,-1,self.head_dim),
                            'bn (hl1 hl2 hl3) d -> bn  hl2 (hl3 hl1) d', 
                            hl1 = self.n_heads//merge_size, hl2 = output_along_token.shape[-2] ,hl3 = merge_size
                            ).reshape(B*nvars,-1,self.head_dim*self.n_heads)
        
        
            output2 = rearrange(output_along_hidden.reshape(B*nvars,-1,self.head_dim),
                            'bn (hl1 hl2 hl3) d -> bn  hl2 (hl3 hl1) d', 
                            hl1 = self.n_heads//merge_size, hl2 = output_along_token.shape[-2] ,hl3 = merge_size
                            ).reshape(B*nvars,-1,self.head_dim*self.n_heads)
        

        output1 = self.norm_post1(output1)
        output1 = output1.reshape(B,nvars, -1, self.n_heads * self.head_dim)
        output2 = self.norm_post2(output2)
        output2 = output2.reshape(B,nvars, -1, self.n_heads * self.head_dim)

        src2 =  self.ff_1(output1)+self.ff_2(output2)
        
        
        src = src + src2
        src = src.reshape(B*nvars, -1, self.n_heads * self.head_dim)
        src = self.norm_attn(src)

        src = src.reshape(B,nvars, -1, self.n_heads * self.head_dim)
        return src
    

def set_parser():
    parser = argparse.ArgumentParser(description='CARD')

    # basic config
    parser.add_argument('--task_name', type=str, default='classification',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=1000, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length')
    parser.add_argument('--pred_len', type=int, default=0, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')

    parser.add_argument('--enc_in', type=int, default=22, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=22, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=22, help='output size')

    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')

    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')
        
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--dp_rank', type=int,default = 8)
    parser.add_argument('--rescale', type=int,default = 1)
    
    parser.add_argument('--fc_dropout', type=float, default=0.3, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.3, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--momentum', type=float, default=0.1, help='momentum')

    
    parser.add_argument('--optimizer', type=str, default='adam')

    parser.add_argument('--local_rank', type=int,default = 0)
    parser.add_argument('--devices_number',type=int,default = 1)
    parser.add_argument('--use_statistic',action='store_true', default=False)
    parser.add_argument('--use_decomp',action='store_true', default=False)
    parser.add_argument('--same_smoothing',action='store_true', default=False)
    parser.add_argument('--warmup_epochs',type=int,default = 0)
    parser.add_argument('--weight_decay',type=float,default = 0)
    parser.add_argument('--merge_size',type=int,default = 2)
    parser.add_argument('--use_untoken',type=int,default = 0)
    parser.add_argument('--pct_start',type=float,default = 0.3)
    
    parser.add_argument('--seg_len', type=int, default=6, help='segment length (L_seg)')
    parser.add_argument('--win_size', type=int, default=2, help='window size for segment merge')
    parser.add_argument('--fix_seed', type = str,default='None')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)
    return args


def get_model():
    args = set_parser()
    args.num_class = 4
    model = Model(args)
    return model


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = get_model().to(device)

    x = torch.randn(16, 1000, 22).to(device).float()
    outputs = model(x)

    # print(model)
    # summary(model, input_size=(1000, 22), batch_size=16)
    print(outputs.shape)
