import torch
from torch import nn
from layers.attention import Attention
from layers.mlp import Mlp, Dinov2SwiGLUFFN
from layers.layer_scale import LayerScale

"""
Transformer block from DINOv2
"""


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        size: str,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        init_values=1,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_class=Attention,
        ffn_layer=Mlp,
    ) -> None:
        super().__init__()
        if size == 'giant':
            ffn_layer = Dinov2SwiGLUFFN
            act_layer = nn.functional.silu
        self.norm1 = norm_layer(dim, eps=1e-6)
        self.attention = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
        )
        self.layer_scale1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

        self.norm2 = norm_layer(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            bias=ffn_bias,
        )
        self.layer_scale2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(self, x: torch.Tensor):

        x = x + self.layer_scale1(self.attention(self.norm1(x)))
        x = x + self.layer_scale2(self.mlp(self.norm2(x)))
        return x