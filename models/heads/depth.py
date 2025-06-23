import torch
from torch import nn


class Head(nn.Module):
    """
    Linear head for depth estimation.
    It concatenates the class token to each patch token.
    """

    def __init__(self, config: dict, output_classes: int = 256):
        super().__init__()
        self.config = config
        self.linear = nn.Linear(2 * config['model']['hidden_size'], output_classes)

    def forward(self, x: torch.Tensor):
        class_tokens, patch_tokens = x[:, 0], x[:, 1:]
        class_tokens = class_tokens.unsqueeze(1).repeat(1,
                                                        (self.config['model']['crop_size'] // self.config['model']['patch_size']) ** 2,
                                                        1)
        x = torch.cat((patch_tokens, class_tokens), dim=-1)
        x = x.reshape((len(x),
                       self.config['model']['crop_size'] // self.config['model']['patch_size'],
                       self.config['model']['crop_size'] // self.config['model']['patch_size'],
                       -1))
        x = x.permute((0, 3, 1, 2))
        x = nn.functional.interpolate(x,
                                      scale_factor=self.config['model']['patch_size'],
                                      mode="bilinear",
                                      align_corners=False)
        x = x.permute((0, 2, 3, 1))
        x = self.linear(x)
        x = x.permute((0, 3, 1, 2))

        return x
