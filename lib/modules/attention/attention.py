from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

from ...utils import shift_dim
from lib.modules.attention.sd_attention import SpatialSelfAttention
from lib.modules.attention.vg_attention import MultiHeadAttention

class DividedSpaceTimeAttention(nn.Module):
    def __init__(self, in_channels, n_head=2):
        """ 時間方向にはAxial Attention 画像方向にはSpatial Attentionを使う それぞれのAttentionでパラメータは共有しない """
        super().__init__()
        self.in_channels = in_channels
        
        self.space_attention = SpatialSelfAttention(in_channels)
        
        kwargs = dict(shape=(0,) * 3, dim_q=in_channels,
                      dim_kv=in_channels, n_head=n_head,
                      n_layer=1, causal=False, attn_type='axial')
        self.axial_attention = MultiHeadAttention(attn_kwargs=dict(axial_dim=-4),
                                                  **kwargs)

    def forward(self, x):
        " x: (B, C, T, H, W)"
        
        b,c,t,h,w = x.shape
        h_ = x.reshape(b*t,c,h,w)
        h_ = self.space_attention(h_)
        s = h_.reshape(b,c,t,h,w)
        
        x_ = shift_dim(x, 1, -1)
        t = self.axial_attention(x_, x_, x_)
        t = shift_dim(t, -1, 1)
        
        x = s + t
        return x