import torch
from torch import nn

"""
Layer scale from DINOv2
"""

class LayerScale(nn.Module):
    def __init__(self, dim,init_values=None) -> None:
        super().__init__()
        if init_values is not None:
            self.lambda1 = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.lambda1
