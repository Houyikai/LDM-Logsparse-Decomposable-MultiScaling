__all__ = ['PatchTST_backbone']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

# from collections import OrderedDict
from layers.PatchTST_layers import *
from layers.RevIN import RevIN
import matplotlib.pyplot as plt
# Cell

class PatchTST_backbone(nn.Module):
    def __init__(self, c_in: int, context_window: int, target_window: int, patch_len: int, stride: int, max_seq_len: Optional[int] = 1024,
                 n_layers: int = 3, d_model=128, n_heads=16, d_k: Optional[int] = None, d_v: Optional[int] = None,
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

        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len)/stride + 1)
        if padding_patch == 'end':  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1

        # Backbone
        self.backbone = TSTiEncoder(c_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                    n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                    attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                    attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                    pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)

        # Head
        self.head_nf = d_model * patch_num
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

    # z: [bs x nvars x seq_len]
    def forward(self, z):
        # norm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0, 2, 1)

        # do patching
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        # z: [bs x nvars x patch_num x patch_len]
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # z: [bs x nvars x patch_len x patch_num]
        z = z.permute(0, 1, 3, 2)

        # model
        # z: [bs x nvars x d_model x patch_num]
        z = self.backbone(z)
        # z: [bs x nvars x target_window]
        z = self.head(z)

        # denorm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0, 2, 1)
        return z

    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout),
                             nn.Conv1d(head_nf, vars, 1)
                             )

# 平头
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

    # x: [bs x nvars x d_model x patch_num]
    def forward(self, x):
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                # z: [bs x d_model * patch_num]
                z = self.flattens[i](x[:, i, :, :])
                # z: [bs x target_window]
                z = self.linears[i](z)
                z = self.dropouts[i](z)
                x_out.append(z)
            # x: [bs x nvars x target_window]
            x = torch.stack(x_out, dim=1)
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x

# Transformer编码器模块（通道独立）
class TSTiEncoder(nn.Module):  # i means channel-independent
    def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):

        super().__init__()

        self.patch_num = patch_num
        self.patch_len = patch_len

        # Input encoding
        q_len = patch_num
        # Eq 1: projection of feature vectors onto a d-dim vector space
        self.W_P = nn.Linear(patch_len, d_model)
        self.seq_len = q_len

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                  pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn)

    # x: [bs x nvars x patch_len x patch_num]
    def forward(self, x) -> Tensor:

        n_vars = x.shape[1]
        # Input encoding
        # x: [bs x nvars x patch_num x patch_len]
        x = x.permute(0, 1, 3, 2)
        # x: [bs x nvars x patch_num x d_model]
        x = self.W_P(x)

        # u: [bs * nvars x patch_num x d_model]
        # u = torch.reshape(x, (x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))
        # u: [bs * nvars x patch_num x d_model]
        u = self.dropout(x + self.W_pos)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        z = self.encoder(u)
        # z: [bs x nvars x patch_num x d_model]
        z = torch.reshape(z, (-1, n_vars, z.shape[-2], z.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        z = z.permute(0, 1, 3, 2)

        return z

# Transformer编码器模块
class TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None,
                 norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                 res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList([AdaptiveEncoderLayer(2, q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                     attn_dropout=attn_dropout, dropout=dropout,
                                                     activation=activation, res_attention=res_attention,
                                                     pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src: Tensor, key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):
        output = src
        scores = None

        # 各层之间是否有残差链接
        if self.res_attention:    
            for mod in self.layers:
                output, scores = mod(
                    output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers:
                output = mod(
                    output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output

class AdaptiveEncoderLayer(nn.Module):
    def __init__(self, split, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        self.split = split
        self.timelayers = nn.ModuleList([TSTEncoderLayer(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout, res_attention=res_attention, pre_norm=pre_norm) for i in range(2)])
        self.crosslayers = nn.ModuleList([CrossEncoderLayer(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout, res_attention=res_attention, pre_norm=pre_norm) for i in range(2)])
                                                       
    # src: [bs x nvars x patch_num x d_model]
    def forward(self, src: Tensor, prev: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None) -> Tensor:
        bs, nvars, seq_len, d_model = src.shape
        # plt.imshow(src[0,6,:,:].detach().cpu().numpy())
        # plt.title('src')
        # plt.show()
        
        # time
        t1 = src[:,:self.split]
        t1 = torch.reshape(t1, (bs*self.split, seq_len, d_model))
        t2 = src[:,self.split:]
        t2 = torch.reshape(t2, (bs*(nvars-self.split), seq_len, d_model))
        t1,_ = self.timelayers[0](t1)
        t1 = t1.view(bs, -1, seq_len, d_model)
        t2,_ = self.timelayers[1](t2)
        t2 = t2.view(bs, -1, seq_len, d_model)

        # plt.imshow(t2[0,0,:,:].detach().cpu().numpy())
        # plt.title('t2')
        # plt.show()
        # # # cross
        c1 = t1.transpose(1, 2)
        c1 = torch.reshape(c1, (-1, self.split, d_model))
        c2 = t2.transpose(1, 2)
        c2 = torch.reshape(c2, (-1, nvars-self.split, d_model))
        c1,_ = self.crosslayers[0](c1, c2)
        c2,_ = self.crosslayers[1](c2, c1)
     
        # # # reshape back
        c1 = c1.view(bs, seq_len, self.split, d_model)
        c1 = c1.transpose(1, 2)
        c2 = c2.view(bs, seq_len, nvars-self.split, d_model)
        c2 = c2.transpose(1, 2)
        # plt.imshow(c2[0,0,:,:].detach().cpu().numpy())
        # plt.title('c2')
        # plt.show()

        src = torch.cat([c1, c2], dim=1)
        return src, None
        

# Transformer编码器层
class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
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

    def forward(self, src: Tensor, prev: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None) -> Tensor:
        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        # Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(
                src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(
                src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
        # Add & Norm
        # Add: residual connection with residual dropout
        src = src + self.dropout_attn(src2)
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

        if self.res_attention:
            return src, scores
        else:
            return src
        
class CrossEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
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

        # Multi-Head attention sublayer
        if self.pre_norm:
            Q = self.norm_attn(Q)
            KV = self.norm_attn(KV)
        # Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(
                Q, KV, KV, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
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

        if self.res_attention:
            return src, scores
        else:
            return src


class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

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

    def forward(self, Q: Tensor, K: Optional[Tensor] = None, V: Optional[Tensor] = None, prev: Optional[Tensor] = None,
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
