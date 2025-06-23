from torch import Tensor, nn
import torch


"""
MLPs used in transformer blocks in DINOv2
"""

class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features= None,
        out_features= None,
        act_layer= nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.activation = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Dinov2SwiGLUFFN(nn.Module):
    def __init__(self, in_features,
            hidden_features=4096,
            act_layer=nn.functional.silu,
            bias=True) -> None:
        super().__init__()
        out_features = in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        self.act_layer = act_layer
        self.weights_in = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.weights_out = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.weights_in(hidden_state)
        x1, x2 = hidden_state.chunk(2, dim=-1)
        hidden = self.act_layer(x1) * x2
        return self.weights_out(hidden)