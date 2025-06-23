import torch
from torch import nn


class Head(nn.Module):
    """
    Linear head for classification.
    Concatenates the class token and the mean of patch tokens
    """
    def __init__(self, config: dict, output_classes: int):
        super().__init__()
        self.linear = nn.Linear(2*config['model']['hidden_size'],
                                output_classes)

    def forward(self, x: torch.Tensor):
        cls_token, patch_tokens = x[:, 0], x[:, 1:]
        linear_input = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1)

        return self.linear(linear_input)
