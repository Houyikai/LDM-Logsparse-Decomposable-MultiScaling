__all__ = ['TbP']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
from torch.nn import LayerNorm
import torch.nn.functional as F
import numpy as np

# from collections import OrderedDict
from layers.PatchTST_layers import *
from layers.PatchTST_layers import gating
from layers.RevIN import RevIN
from einops import rearrange, repeat
from math import sqrt
import matplotlib.pyplot as plt

show = False

# Transformer-based Predictor
class TbP(nn.Module):
    def __init__(self, c_in: int, context_window: int, target_window: int, patch_len: int, stride: int, max_seq_len: Optional[int] = 1024,
                 n_layers: int = 3, d_model=128, n_heads=16, factor=10, d_k: Optional[int] = None, d_v: Optional[int] = None,
                 d_ff: int = 256, norm: str = 'BatchNorm', attn_dropout: float = 0., dropout: float = 0., act: str = "gelu", key_padding_mask: bool = 'auto',
                 padding_var: Optional[int] = None, attn_mask: Optional[Tensor] = None, res_attention: bool = True, pre_norm: bool = False, store_attn: bool = False,
                 pe: str = 'zeros', learn_pe: bool = True, fc_dropout: float = 0., head_dropout=0, padding_patch=None,
                 pretrain_head: bool = False, head_type='flatten', individual=False, revin=True, affine=True, subtract_last=False,
                 verbose: bool = False, **kwargs):

        super().__init__()
        # RevIn
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(
                c_in, affine=affine, subtract_last=subtract_last)
        trendfitting = context_window == patch_len
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len)/stride + 1)
        if padding_patch == 'end':  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1
        
        self.backbone = MultiTSTEncoder(c_in, trendfitting, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                    n_layers=n_layers, d_model = d_model, n_heads=n_heads, factor=factor, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                    attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                    attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                    pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)

        # Head
        self.head_nf = d_model * (patch_len + patch_num)
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual
        
        if self.pretrain_head:
            # custom head passed as a partial func with all its kwargs
            self.head = self.create_pretrain_head(
                self.head_nf, c_in, fc_dropout)
        elif head_type == 'flatten':
            self.head = Flatten_Head(
                self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)

    # z: [bs , seq_len, nvars]
    def forward(self, z):
        # norm
        if self.revin:
            z = self.revin_layer(z, 'norm')
        z = z.permute(0, 2, 1)

        # do patching
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        
        # z: [bs x nvars x patch_num x patch_len]
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        
        
        # model
        # z: [bs x nvars x patch_len x patch_num]
        z = z.permute(0, 1, 3, 2)
        # z: [bs x nvars x d_model x patch_num] 
        z = self.backbone(z)
        # d_model at last dim
        z = z.permute(0, 1, 3, 2)
   
        # z: [bs x nvars x target_window]
        z = self.head(z)

        z = z.permute(0, 2, 1)
        # denorm
        if self.revin:
            z = self.revin_layer(z, 'denorm')

        return z

    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout),
                             nn.Conv1d(head_nf, vars, 1)
                             )

class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
        
                z = self.flattens[i](x[:, i, :, :])
  
                z = self.linears[i](z)
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)
        else:
            x = self.flatten(x)
            if show:
                plt.plot(x[0,0,:].detach().cpu().numpy())
                plt.show()
            x = self.linear(x)
            x = self.dropout(x)
        return x

class MultiTSTEncoder(nn.Module): 
    def __init__(self, c_in, trend_fitting, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, factor=10, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):

        super().__init__()

        self.patch_num = patch_num
        self.patch_len = patch_len
        # Input encoding
        q_len = patch_num
        # Eq 1: projection of feature vectors onto a d-dim vector space
        self.W_L = nn.Linear(patch_len, d_model)
        self.W_S = nn.Linear(patch_num, d_model)
        self.seq_len = q_len
        self.d_model = d_model
        # Positional encoding
        self.W_posL = positional_encoding(pe, learn_pe, patch_num, d_model)
        self.W_posS = positional_encoding(pe, learn_pe, patch_len, d_model)
        # Residual dropout
        self.dropout_embed = nn.Dropout(0.3)
   
        # Encoder
        self.layers = nn.ModuleList([iEncoderLayer(d_model=d_model, n_heads=n_heads, patch_num=patch_num, patch_len=patch_len, factor=factor, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                     attn_dropout=attn_dropout, dropout=dropout,
                                                     act=act, res_attention=res_attention,
                                                     pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

        self.n_layers = n_layers

    # x: [bs x nvars x patch_len x patch_num]
    def forward(self, x: Tensor, key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None) -> Tensor:

        n_vars = x.shape[1]
        x = x.permute(0, 1, 3, 2)
        
        # embedding
        # x: [bs x nvars x patch_num x d_model]
        xl = self.W_L(x)
        xs = x.permute(0,1,3,2)
        # x: [bs x nvars x patch_len x d_model]
        xs = self.W_S(xs)

        xl = self.dropout_embed(xl + self.W_posL)
        xs = self.dropout_embed(xs + self.W_posS)
        # encoding
        # outputl, outputs = torch.reshape(xl,(-1, xl.shape[-2], xl.shape[-1])), torch.reshape(xs,(-1, xs.shape[-2], xs.shape[-1])) 
        outputl, outputs = xl, xs 
        if self.res_attention:    
            for mod in self.layers:
                outputl, outputs = mod(
                    outputl, outputs, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            for mod in self.layers:
                output = mod(
                    output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

        # z: [bs x nvars x (patch_num + patch_len) x d_model]
        z = torch.concat((outputl, outputs), dim=-2)
        
        z = torch.reshape(z, (-1, n_vars, z.shape[-2], z.shape[-1]))
        z = z.permute(0, 1, 3, 2)

        return z

class iEncoderLayer(nn.Module):
    def __init__(self, patch_num, patch_len, d_model=128, n_heads=16, factor=10, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False):
        super().__init__()
       
        self.res_attention = res_attention
        self.self_attnL = selfEncoderLayer(d_model=d_model, n_heads=n_heads, seq_len=patch_num, factor=factor, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                     attn_dropout=attn_dropout, dropout=dropout,
                                                     activation=act, res_attention=res_attention,
                                                     pre_norm=pre_norm, store_attn=store_attn)

        self.self_attnS = selfEncoderLayer(d_model=d_model, n_heads=n_heads, seq_len=patch_len, factor=factor, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                     attn_dropout=attn_dropout, dropout=dropout,
                                                     activation=act, res_attention=res_attention,
                                                     pre_norm=pre_norm, store_attn=store_attn)

        self.cross_attnL = crossEncoderLayer(
            d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, res_attention=res_attention)
        
        self.cross_attnS = crossEncoderLayer(
            d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, res_attention=res_attention)
     
        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
      
    # srcL: [bs x nvars x patch_num x d_model]
    # srcS: [bs x nvars x patch_len x d_model]
    def forward(self, srcL: Tensor, srcS: Tensor, prev: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None) -> Tensor:
 
        # time
        src2_L, _, _ = self.self_attnL(srcL, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        src2_S, _, _ = self.self_attnS(srcS, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        
        src2_L,_ = self.cross_attnL(src2_L, src2_S, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        src2_S,_ = self.cross_attnS(src2_S, src2_L, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # print('interP:',src2_L.shape, 'intraP:',src2_S.shape)

        # src2_S = torch.zeros_like(src2_S)
        # src2_L = torch.zeros_like(src2_L)
        
        return src2_L, src2_S

class crossEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(
            d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
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
            self.norm_ffn = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(self, Q: Tensor, KV: Tensor, prev: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None) -> Tensor:
        nvars = Q.shape[1]
        Q = torch.reshape(Q, (-1, Q.shape[-2], Q.shape[-1]))
        KV = torch.reshape(KV, (-1, KV.shape[-2], KV.shape[-1]))
        # Multi-Head attention sublayer
        if self.pre_norm:
            Q = self.norm_attn(Q)
            KV = self.norm_attn(KV)

        # Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(
                Q, KV, KV, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(
                Q, KV, KV, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
        # Add & Norm
        # Add: residual connection with residual dropout
        src = Q + self.dropout_attn(src2)
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        # Position-wise Feed-Forward
        src2 = self.ff(src)
        # Add & Norm
        # Add: residual connection with residual dropout
        src = src + self.dropout_ffn(src2)
        if not self.pre_norm:
            src = self.norm_ffn(src)
        
        src = src.reshape(-1, nvars, src.shape[-2], src.shape[-1])
        if self.res_attention:
            return src, scores
        else:
            return src

class selfEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, seq_len, factor, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model


        # multi-dimensional attention
        self.res_attention = res_attention
        self.self_attn = _SpatialTemporalAttention(
            d_model, n_heads, seq_len, factor, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
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
            self.norm_ffn = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    # src: [bs x nvars x patch_num x d_model]
    def forward(self, src: Tensor, prev_time: Optional[Tensor] = None, prev_space: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None) -> Tensor:
        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)

        # Multi-Head attention
        if self.res_attention:
            src2, attn, scores_time, scores_space  = self.self_attn(
                src, prev_time, prev_space, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(
                src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn

        bs, nvars, patch_num, d_model = src.shape

        # Add & Norm
        # Add: residual connection with residual dropout
        src = src + self.dropout_attn(src2)
        if not self.pre_norm:
            src = self.norm_attn(src.reshape(bs, nvars* patch_num, d_model)).reshape(bs, nvars, patch_num, d_model)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src.reshape(bs, nvars* patch_num, d_model)).reshape(bs, nvars, patch_num, d_model)

        # spacetime-wise Feed-Forward
        src2 = self.ff(src.reshape(bs, nvars* patch_num, d_model)).reshape(bs, nvars, patch_num, d_model)
    
        # Add & Norm
        # Add: residual connection with residual dropout
        src = src + self.dropout_ffn(src2)
        if not self.pre_norm:
            src = self.norm_ffn(src.reshape(bs, nvars* patch_num, d_model)).reshape(bs, nvars, patch_num, d_model)

        if self.res_attention:
            return src, scores_time, scores_space
        else:
            return src

class _SpatialTemporalAttention(nn.Module):
    def __init__(self, d_model, n_heads, seq_len, factor, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()

        self.d_model = d_model
        self.res_attention = res_attention
        self.ct_attn = _MultiheadAttention(
            d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=proj_dropout, res_attention=res_attention)
        # self.dim_sender = _MultiheadAttention(
        #     d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=proj_dropout, res_attention=res_attention)
        # self.dim_receiver = _MultiheadAttention(
        #     d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=proj_dropout, res_attention=res_attention)

        # self.router = nn.Parameter(torch.randn(seq_len, factor, d_model))

    # src: [bs x nvars x patch_num x d_model]
    def forward(self, src: Tensor, prev_time: Optional[Tensor] = None, prev_space: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):
  
        bs = src.shape[0]
        nvars = src.shape[1]
        seq_len = src.shape[2]
        src_time = torch.reshape(src, (bs * nvars, seq_len, self.d_model))
        src_space = torch.reshape(src.permute(0,2,1,3), (bs*seq_len, nvars, self.d_model))

        # multi-dimensional embedding
        if self.res_attention:
            
            src2_time, attn_time, scores_time = self.ct_attn(
                src_time, src_time, src_time, prev = prev_time, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            # [bs x nvars x patch_num x d_time]
            src2_time = torch.reshape(src2_time, (bs, nvars, seq_len, self.d_model))


            # dim_send = src_space
            # batch_router = [L,c,d] ， c is the number of routers
            # batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat = bs)
            # dim_buffer, attn_1, scores_1 = self.dim_sender(
            #     batch_router, dim_send, dim_send, prev = prev_space, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            # dim_receive, attn_2, scores_2 = self.dim_receiver(
            #     dim_send, dim_buffer, dim_buffer, prev = prev_space, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            # # [bs x nvars x patch_num x d_space]
            # src2_space = torch.reshape(dim_receive, (bs, seq_len, nvars, self.d_model)).permute(0,2,1,3)
            scores_space = None
  
            src2 = src2_time

            attn_time = torch.reshape(attn_time, (bs, nvars, attn_time.shape[1], attn_time.shape[2], attn_time.shape[3]))
            attn = None
            scores_time

            return src2, attn, scores_time, scores_space
            
        else:
            src2_time, attn_time = self.ct_attn(
                src_time, src_time, src_time, prev_time, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            src2_space, attn_space = self.cs_attn(
                src_space, src_space, src_space, prev_space, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            src2 = torch.cat((src2_time, src2_space), dim=-1)
            attn = torch.cat((attn_time, attn_space), dim=-1)
            return src2, attn

class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        # d_k = d_model // n_heads if d_k is None else d_k
        # d_v = d_model // n_heads if d_v is None else d_v
        d_k = d_v = d_model // n_heads 
        

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(
            d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(
            nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))
        
    def forward(self, Q: Tensor, K: Optional[Tensor] = None, V: Optional[Tensor] = None, sigma: Optional[Tensor] = None, prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):
        bs = Q.size(0)
        if K is None:
            K = Q
        if V is None:
            V = Q

        # Linear (+ split in multiple heads)
        # q_s    : [bs x n_heads x max_q_len x d_k]
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        k_s = self.W_K(K).view(bs, -1, self.n_heads,
                               self.d_k).permute(0, 2, 3, 1)
        # v_s    : [bs x n_heads x q_len x d_v]
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(
                q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(
                q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(
            bs, -1, self.n_heads * self.d_v)  # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights

class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(
            head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q: Tensor, k: Tensor, v: Tensor, prev: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        # attn_scores : [bs x n_heads x max_q_len x q_len]
        attn_scores = torch.matmul(q, k) * self.scale

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None:
            attn_scores = attn_scores + prev

        # Attention mask (optional)
        # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
        
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        # mask with shape [bs x q_len] (only when max_w_len == q_len)
        if key_padding_mask is not None:
            attn_scores.masked_fill_(
                key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)  

        # compute the new values given the attention weights
        # output: [bs x n_heads x max_q_len x d_v]
        output = torch.matmul(attn_weights, v)

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights


