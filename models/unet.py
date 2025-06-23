import torch
from torch import nn
from utils.layer_norm import LayerNorm


class UNet(nn.Module):
    """
    UNet used in a ViT-UNet hybrid.
    It is a 3 blocks-deep UNet merged through a linear layer with the output of a ViT.

    If a preferred bias initialization exists, it can be provided as the 'initialization' argument.
    """

    def __init__(self,
                 encoder: nn.Module,
                 config,
                 num_blocks: int = 3,
                 num_classes: int = 1,
                 biased_initialization: bool = True,
                 initialization: torch.Tensor = None):
        """
        :param encoder: ViT to use for the hybrid
        :param config: ViT's configuration file
        :param num_blocks: number of convolutional blocks
        :param num_classes: number of channels for the UNet's output
        :param biased_initialization: True for replacing the bias for the UNet's last layer
        :param initialization: tensor to replace the UNet's last layer with
        """
        super().__init__()
        layers = [nn.Conv2d(3, 2 ** 4, 3, padding='same')]
        for _ in range(num_blocks):
            layers.append(LayerNorm(2 ** (4 + _)))
            layers.append(nn.GELU())
            layers.append(nn.Conv2d(2 ** (4 + _), 2 ** (5 + _), 3, padding='same'))
            layers.append(LayerNorm(2 ** (5 + _)))
            layers.append(nn.GELU())
            layers.append(nn.Conv2d(2 ** (5 + _), 2 ** (5 + _), 2, 2))
        self.downsampling = nn.ModuleList(layers)
        self.config = config
        self.encoder = encoder
        self.norm_latent = LayerNorm(2 ** (4 + num_blocks))
        self.merger = nn.Conv2d(2 * self.config['model']['hidden_size'], 2 ** (4 + num_blocks), 1)

        self.norm_output = LayerNorm(2 ** 5)
        self.output_layer = nn.Conv2d(2 ** 5, num_classes, 1)
        if biased_initialization:
            state_dict = self.output_layer.state_dict()
            if initialization is None:
                state_dict['bias'][0] = 3 * torch.abs(state_dict['bias'][0])
                state_dict['bias'][1:] = - 1 * torch.abs(state_dict['bias'][1:])
            else:
                state_dict['bias'] = initialization
            self.output_layer.load_state_dict(state_dict)

        upsampling_layers = []
        for _ in reversed(range(num_blocks)):
            upsampling_layers.append(LayerNorm(2 ** (6 + _)))
            upsampling_layers.append(nn.GELU())
            upsampling_layers.append(nn.Conv2d(2 ** (6 + _), 2 ** (4 + _), 3, padding='same'))
            upsampling_layers.append(LayerNorm(2 ** (4 + _)))
            upsampling_layers.append(nn.GELU())
            upsampling_layers.append(nn.ConvTranspose2d(2 ** (4 + _), 2 ** (4 + _), 2, 2))
        self.upsampling_layers = nn.ModuleList(upsampling_layers)

    def forward(self, x: torch.Tensor):
        batch_size, c, h, w = x.shape
        with torch.no_grad():
            embedding = self.encoder(x.clone())
        blocks_outputs = []
        for l, layer in enumerate(self.downsampling):
            x = layer(x)
            if isinstance(layer, nn.Conv2d) and (l % 6 == 0):
                blocks_outputs.append(x.clone())

        cls_token, patch_token = embedding[:, 0], embedding[:, 1:]
        patch_token = patch_token.reshape((len(patch_token),
                                           h // self.config['model']['patch_size'],
                                           w // self.config['model']['patch_size'],
                                           -1))

        patch_token = patch_token.permute((0, 3, 1, 2))
        patch_token = nn.functional.interpolate(patch_token,
                                                size=(blocks_outputs[-1].shape[-1], blocks_outputs[-1].shape[-1]),
                                                mode="bilinear",
                                                align_corners=False)
        cls_token = cls_token.view((len(cls_token), self.config['model']['hidden_size'], 1, 1))
        cls_token = cls_token.repeat((1, 1, patch_token.shape[-1], patch_token.shape[-1]))
        embedding = torch.cat((patch_token, cls_token), dim=1)
        x = self.merger(embedding)
        x = torch.cat((x, blocks_outputs[-1]), dim=1)

        l = 2
        for layer in self.upsampling_layers:
            x = layer(x)
            if isinstance(layer, nn.ConvTranspose2d):
                x = torch.cat((x, blocks_outputs[-l]), dim=1)
                l += 1

        x = self.norm_output(x)
        x = nn.functional.gelu(x)
        x = self.output_layer(x)

        return x
