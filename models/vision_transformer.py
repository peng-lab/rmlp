import torch
from torch import nn
from layers.block import Block
from layers.embeddings import Embeddings


class Encoder(nn.Module):
    """
    ViT architecture from DINOv2.
    iBOT loss in included if an ibot_mask and mask_token are provided during forward.
    """
    def __init__(
        self,
        config,
        embed_layer: nn.Module = Embeddings,
        act_layer=nn.GELU,
        block_fn: nn.Module = Block
    ):
        super().__init__()
        norm_layer = nn.LayerNorm
        self.config = config
        self.embeddings = embed_layer(config)

        blocks_list = [
            block_fn(
                dim=config['model']['hidden_size'],
                size=config['model']['size'],
                num_heads=config['model']['num_attention_heads'],
                mlp_ratio=config['model']['mlp_ratio'],
                qkv_bias=config['model']['qkv_bias'],
                norm_layer=norm_layer,
                act_layer=act_layer
            )
            for i in range(config['model']['num_hidden_layers'])
        ]
        self.transformers = nn.ModuleList(blocks_list)
        self.norm = norm_layer(config['model']['hidden_size'], eps=1e-6)

    def forward(self, x: torch.Tensor,
                ibot_mask: torch.Tensor = None,
                mask_token: torch.Tensor = None):
        x = self.embeddings(x)
        if ibot_mask is not None:
            x = torch.where(ibot_mask, mask_token, x)
        for blk in self.transformers:
            x = blk(x)
        x = self.norm(x)

        return x
