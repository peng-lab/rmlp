import argparse
import torch
import yaml
from train.dst import DST
from models.heads.depth import Head
from torch import nn
from models.vision_transformer import Encoder
from models.unet import UNet
from transformers import Dinov2Model
from data.nyu import Loader, TestLoader


def rmse(inputs: torch.Tensor, target: torch.Tensor):
    x = torch.argmax(inputs, dim=1).type(torch.float32).to(target.device)
    return torch.sqrt(nn.MSELoss()(x, target.type(torch.float32)))


class FocalLoss(nn.Module):
    def __init__(self, num_classes: int, gamma: float = 2):
        super().__init__()
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-8):
        inputs = torch.nn.functional.softmax(inputs, dim=1)
        targets = torch.nn.functional.one_hot(targets, num_classes=self.num_classes).to(inputs.device).permute(
            (0, 3, 1, 2))
        p_t = torch.where(targets == 1, inputs, 1 - inputs)
        focal = -torch.mul((1 - p_t) ** self.gamma, torch.log(p_t + smooth)).mean()
        return focal


class LastHiddenState(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, x: torch.Tensor):
        return self.encoder(x).last_hidden_state


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x


def main(config_path: str):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    encoder = Encoder(config=config)
    if config['checkpoint']['pretrain_path'] is not None:
        encoder.load_state_dict(torch.load(config['checkpoint']['pretrain_path']))
    if config['checkpoint']['dinov2_checkpoint']:
        encoder = Dinov2Model.from_pretrained(f"facebook/dinov2-{config['model']['size']}")
        encoder = LastHiddenState(encoder)

    if config['model']['use_unet']:
        encoder = UNet(encoder=encoder,
                       config=config,
                       num_classes=config['model']['num_classes'],
                       biased_initialization=False)

    head = Head(config, output_classes=config['model']['num_classes'])
    loader = Loader(config['data']['data_path'])
    test_loader = TestLoader(config['data']['data_path'])

    model = DST(encoder=encoder,
                head=head,
                loss_function=FocalLoss(num_classes=config['model']['num_classes']),
                test_metrics={'cross_entropy': nn.CrossEntropyLoss(),
                              'rmse': rmse},
                loader=loader,
                test_loader=test_loader,
                save_path=f'{config["checkpoint"]["save_path"]}/checkpoints',
                name=config['checkpoint']['name'],
                unet=config['model']['use_unet'])

    print('Begin training', config['checkpoint']['name'])

    model.do_train(epochs=config['training']['epochs'],
                   steps_per_epoch=config['training']['steps_per_epoch'],
                   steps_per_val=config['training']['steps_per_epoch_val'],
                   batch_size=config['training']['batch_size'],
                   lr=float(config['training']['lr']))

    model.head.load_state_dict(torch.load(f'{model.save_path}/{model.name}_head_best.pt'))
    model.encoder.load_state_dict(torch.load(f'{model.save_path}/{model.name}_encoder_best.pt'))

    model.do_evaluate(f'{config["checkpoint"]["save_path"]}/test_results', task='depth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate the model with a specified config file.")
    parser.add_argument('config_path', type=str, help="Path to the configuration YAML file")

    args = parser.parse_args()

    main(args.config_path)
