import torch
from torch import nn
import math

"""
Tokenizer from DINOv2
"""

class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config['model']['hidden_size']
        self.cls_token = nn.Parameter(torch.normal(0,1,(1,1,self.hidden_size)))
        self.mask_token = nn.Parameter(torch.normal(0,1,(1,self.hidden_size)))
        self.patch_embeddings = nn.Conv2d(3, self.hidden_size, kernel_size=config['model']['patch_size'],
                                          stride=config['model']['patch_size'])
        self.position_embeddings = nn.Parameter(torch.zeros((1,1370,self.hidden_size)))
        self.config = config

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1
        if num_patches == num_positions and height == width:
            return self.position_embeddings
        class_pos_embed = self.position_embeddings[:, 0]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = embeddings.shape[-1]
        height = height // self.config['model']['patch_size']
        width = width // self.config['model']['patch_size']
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        height, width = height + 0.1, width + 0.1
        patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed,
                scale_factor=(float(height / math.sqrt(num_positions)), float(width / math.sqrt(num_positions))),
                mode="bicubic",
                align_corners=False, )

        if int(height) != patch_pos_embed.shape[-2] or int(width) != patch_pos_embed.shape[-1]:
            raise ValueError("Width or height does not match with the interpolated position embeddings")
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(self, x):
        batch_size, _, height, width = x.shape
        embeddings = self.patch_embeddings(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)

        return embeddings
