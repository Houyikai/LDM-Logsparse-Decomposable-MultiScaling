__all__ = ['MTE']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
import math
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from layers.ppr import ppr
import matplotlib.ticker as ticker
import pandas as pd
ACTIVATIONS = ["ReLU", "Softplus", "Tanh", "SELU", "LeakyReLU", "PReLU", "Sigmoid"]
POOLING = ["MaxPool1d", "AvgPool1d"]


class Model(nn.Module):
    def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        
        super().__init__()
       
        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        
        individual = configs.individual
        
        revin = configs.revin

        affine = configs.affine
        subtract_last = configs.subtract_last
        
        decomposition = configs.decomposition
        kernel_size = configs.kernel_size
        factor = configs.factor
        self.input_len = context_window
        self.period_list = configs.period_list
        print(self.period_list)
        self.bs = configs.batch_size
        self.pred_len = configs.pred_len
        self.c_in = configs.enc_in
        # model
        self.decomposition = decomposition
        # period = 24

        self.interpolation_mode = 'linear'
        self.target_window = target_window  
        l = context_window
        h = target_window

        self.seq_lens = []
        self.pred_lens = []
        self.t_kernal = 2 # pooling kernal size, bigger means smaller kernal
        self.t_stride = 2 # pooling stride, bigger means smaller stride
        for period in [2] + self.period_list:
          kernel_size = int(period/self.t_kernal)
          stride = int(period/self.t_stride)
          l_out = math.ceil((l - kernel_size)/stride+1) 
          h_p = h/l * l_out
          self.seq_lens.append(l_out)
          self.pred_lens.append(int(h_p))

        print("input and output length per layers:",self.seq_lens, self.pred_lens)
        self.pooling_period_len = [int(period/l*seq_len) for period, seq_len in zip(self.period_list+[context_window], self.seq_lens)]
        print("period_len per layer:", self.pooling_period_len)

        # max_period_num = configs.period_num
        max_period_num = configs.period_num

        self.layers = nn.ModuleList([ppr(c_in=c_in, context_window=min([seq_len,int(period/l*seq_len * max_period_num)]), target_window=pred_len, patch_len=int(period/l*seq_len), stride=int(period/l*seq_len), 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, factor=factor, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_period = None,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs) for period, seq_len, pred_len in zip(self.period_list + [context_window], self.seq_lens, self.pred_lens)])
        # for trunc
        self.input_lens = [min([seq_len,int(period/l*seq_len * max_period_num)]) for period, seq_len in zip(self.period_list+[context_window], self.seq_lens)]

        self.decompose = Hdecompose(period_list=self.period_list) # last period is the trend
        self.pool = Hpool(period_list=self.period_list, t_kernal=self.t_kernal, t_stride=self.t_stride)

  
       
        
    # x: [Batch, Input length, Channel]
    def forward(self, x):
        
        if isinstance(self.period_list, list):
            
            x = self.decompose(x)
           
            x = self.pool(x)
            pred_list = []
            start = 0

            for i, layer in enumerate(self.layers):

                input_i = x[:,:,start:start+ self.seq_lens[i]].transpose(-1,-2)
                input_i = input_i[:,-self.input_lens[i]:,:] # trunc
                h_i = layer(input_i) 
                h_i = F.interpolate(
                    h_i.transpose(-1,-2), size=self.target_window, mode=self.interpolation_mode
                ).transpose(-1,-2)  
                start += self.seq_lens[i]
                pred_list.append(h_i)
                
            # IN_last = input_i[:,-1,:]
            # OUT_last = h_i[:,-1,:]
            # h_i += IN_last.unsqueeze(1) - OUT_last.unsqueeze(1)
            z = torch.stack(pred_list, dim=-1).sum(-1)
        else:
            z = self.backbone(x)
        
        return z

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2+1, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Hdecompose(nn.Module):
    def __init__(self, period_list=[4, 96]):
        super().__init__()
        self.period_list = period_list
    def forward(self, x): # x: [bs, l, d]
        season_list = []
        trend_init = x
        for i in range(len(self.period_list)):
            decomp = series_decomp(self.period_list[i])
            season, trend = decomp(trend_init)
            trend_init = trend
            season_list.append(season)
        season_list.append(trend)  
        return torch.stack(season_list, dim=1)
    
# z: [bs, part, d, l]
def hpool(z, period_list, t_kernal=2, t_stride=2):
    """
    Hierarchical pooling layer
    """
    pooling_list = []
    for i in range(z.shape[1]):
        if i<len(period_list):
            pooling_layer = getattr(nn, POOLING[1])(
                kernel_size=int(period_list[i]/t_kernal), stride=int(period_list[i]/t_stride), ceil_mode=True
            )
            pooling_list.append(pooling_layer(z[:,i,:,:]))
    return torch.cat(pooling_list, dim=-1)

class Hpool(nn.Module):

    def __init__(self, period_list=[4, 96], t_kernal=2, t_stride=2):
        super().__init__()
        self.t_kernal = t_kernal
        self.t_stride = t_stride
        self.period_list = [2] + period_list
    def forward(self, x): # x: [bs, p, l, d]
        # result: [bs, d, p]
        return hpool(x.transpose(-1,-2), self.period_list, self.t_kernal, self.t_stride)

