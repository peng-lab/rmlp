import torch
from torch import nn
import numpy as np


class Head(torch.nn.Module):
    """
    Linear head for semantic segmentation.
    It segments using the patch tokens.
    """

    def __init__(self, config: dict, output_classes: int = 1):
        super().__init__()

        self.config = config
        self.classifier = torch.nn.Conv2d(config['model']['hidden_size'], output_classes, (1, 1))

    def forward(self, x: torch.Tensor):
        x = x[:,1:].reshape((len(x),
                       int(np.sqrt(x.shape[1])),
                       int(np.sqrt(x.shape[1])),
                       self.config['model']['hidden_size']))
        x = x.permute((0, 3, 1, 2))
        x = torch.stack([nn.functional.interpolate(x[_].unsqueeze(0),
                                                   scale_factor=self.config['model']['patch_size'],
                                                   mode="bilinear",
                                                   align_corners=False)[0] for _ in range(len(x))])

        x = self.classifier(x)

        return x
