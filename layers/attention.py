from torch import nn
import torch
import math

"""
Attention mechanism from DINOv2
"""


class attention(nn.Module):
    def __init__(self, dim, bias=True, num_heads=8, attn_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.attention_head_size = dim // num_heads
        self.all_head_size = self.num_heads * self.attention_head_size
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.query = nn.Linear(dim, dim, bias=bias)
        self.key = nn.Linear(dim, dim, bias=bias)
        self.value = nn.Linear(dim, dim, bias=bias)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        key_layer = self.transpose_for_scores(self.key(x))
        value_layer = self.transpose_for_scores(self.value(x))
        query_layer = self.transpose_for_scores(self.query(x))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer


class Output(nn.Module):
    def __init__(self, dim, bias=True, proj_drop=0.0):
        super().__init__()
        self.dense = nn.Linear(dim, dim, bias=bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        x = self.dense(x)
        return self.proj_drop(x)


class Attention(nn.Module):
    def __init__(self, dim: int,
                 num_heads: int = 8,
                 qkv_bias: bool = True,
                 proj_bias: bool = True,
                 attn_drop: float = 0.0,
                 proj_drop: float = 0.0):
        super().__init__()

        self.attention = attention(dim, bias=qkv_bias, num_heads=num_heads, attn_drop=attn_drop)
        self.output = Output(dim, bias=proj_bias, proj_drop=proj_drop)

    def forward(self, x):
        x = self.attention(x)
        x = self.output(x)
        return x
