import os
import torch
from torch import nn
from models.vision_transformer import Encoder
from loss.dino_loss import DINOLoss
from loss.koleo_loss import KoLeoLoss
import numpy as np
from tqdm import trange
from layers.dino_head import _build_mlp
from transformers import Dinov2Model
from typing import Callable
import yaml
from data.loaders import Loader
import argparse


class TrainingModel(nn.Module):
    """
    Class implementing the contrastive learning algorithm
    """

    def __init__(self,
                 config_path: str,
                 train_loader: Callable,
                 val_loader: Callable,
                 device: torch.device = torch.device('cpu'),
                 ):
        """
        :param config_path: path where to fing the .yaml file with the general configuration
        :param train_loader: callable from where to get inputs for training
        :param val_loader: callable from where to get inputs for valitation
        :param device: torch device for the model
        """
        super().__init__()
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
        self.device = device

        self.student = Encoder(config=self.config)
        self.teacher = Encoder(config=self.config)
        for param in self.teacher.parameters():
            param.requires_grad = False
        if self.config["model"]["use_dinov2"]:
            dino_model = Dinov2Model.from_pretrained(f"facebook/dinov2-{self.config['model']['size']}")
            self.student.embeddings.patch_embeddings.load_state_dict(
                dino_model.embeddings.patch_embeddings.projection.state_dict())
            self.student.transformers.load_state_dict(dino_model.encoder.layer.state_dict())
            self.student.embeddings.cls_token = nn.Parameter(dino_model.embeddings.cls_token.clone())
            self.student.embeddings.mask_token = nn.Parameter(dino_model.embeddings.mask_token.clone())
            self.student.embeddings.position_embeddings = nn.Parameter(
                dino_model.embeddings.position_embeddings.clone())
        if self.config['checkpoint']['pretrain_path'] is not None:
            self.student.load_state_dict(self.config['checkpoint']['pretrain_path'])
        self.teacher.load_state_dict(self.student.state_dict())

        self.head_s = _build_mlp(inp_dim=self.config['model']['hidden_size'],
                                 out_dim=int(2 * self.config['model']['hidden_size'] / 3) ** 2,
                                 hidden_dim=4 * self.config['model']['hidden_size'],
                                 bottleneck_dim=int(2 * self.config['model']['hidden_size'] / 3),
                                 rmlp=self.config['model']['rmlp'])

        self.head_t = _build_mlp(inp_dim=self.config['model']['hidden_size'],
                                 out_dim=int(2 * self.config['model']['hidden_size'] / 3) ** 2,
                                 hidden_dim=4 * self.config['model']['hidden_size'],
                                 bottleneck_dim=int(2 * self.config['model']['hidden_size'] / 3),
                                 rmlp=self.config['model']['rmlp'])

        self.dino_loss = DINOLoss(int(2 * self.config['model']['hidden_size'] / 3) ** 2)
        self.ibot_loss = DINOLoss(int(2 * self.config['model']['hidden_size'] / 3) ** 2)
        self.koleo_loss = KoLeoLoss()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.teacher_temp = self.config['training']['tpt_0']
        self.student_temp = self.config['training']['tps_0']
        self.center_momentum = self.config['training']['center_momentum']
        self.teacher_momentum = self.config['training']['teacher_momentum']
        self.dino_coef = self.config['training']['dino_coef']
        self.ibot_coef = self.config['training']['ibot_coef']
        self.koleo_coef = self.config['training']['koleo_coef']
        self.history = [[], []]

    def get_class_token(self,
                        features: torch.Tensor,
                        crop_size: tuple,
                        patch_size: int,
                        hidden_size: int):
        """
        :param features: vit's output
        :param crop_size: size of the input
        :param patch_size: patch size from the model
        :param hidden_size: token size from the model
        :return: class token for dino loss
        """
        features = features.reshape((len(features), -1,
                                     1 + (crop_size[0] // patch_size) ** 2,
                                     hidden_size))
        class_tokens = features[..., 0, :]
        return class_tokens

    def update_teacher(self):
        """
        Function for updating teacher's parameters
        :return: None
        """
        teacher_params = self.teacher.state_dict()
        student_params = self.student.state_dict()
        for key in teacher_params.keys():
            teacher_params[key] = self.teacher_momentum * teacher_params[key] + (1 - self.teacher_momentum) * \
                                  student_params[key].clone()
        self.teacher.load_state_dict(teacher_params)

    def save_models(self,
                    val_loss: torch.Tensor,
                    early_stop: int,
                    stage: int,
                    thr: float = 0.1):
        """
        :param val_loss: validation loss from epoch
        :param early_stop: current amount of epochs without improvement in validation loss
        :param stage: stage of training. 0 for warmup, 1 for regular training
        :param thr: minimum improvement for to consider the model improved
        :return: updated early_stop value depending on model's improvement
        """
        if (val_loss.item() + thr) < self.best:
            self.best = val_loss.item()
            early_stop = 0
            torch.save(self.student.state_dict(),
                       f'{self.config["checkpoint"]["save_path"]}/{self.config["checkpoint"]["name"]}/student_{stage}_best.pt')
        else:
            early_stop += 1
        return early_stop

    @torch.no_grad()
    def create_memory_bank(self,
                           loader: Callable,
                           crop_size: tuple,
                           memory_bank_size: int = 1024):
        """
        Function for creating a memory bank
        :param loader: train or validation loader
        :param crop_size: size of global view of images
        :param memory_bank_size: amount of elements in memory bank
        :return: None. It stores the memory bank as a class attribute
        """
        mb = []
        for _ in range(memory_bank_size):
            s_batch, t_batch = loader.create_dino_batch(batch_size=1,
                                                        repeats=(1, 1),
                                                        crop_size=((self.config['training']['local_size'],
                                                                    self.config['training']['local_size']),
                                                                   crop_size))

            s = self.student(s_batch.to(self.device)[0])
            mb.append(s.clone()[0])

        self.memory_bank = torch.stack(mb)

    def training_step(self,
                      batch_size: int,
                      crop_size: tuple,
                      repeats: tuple):
        """
        :param batch_size: batch size
        :param crop_size: size for global views
        :param repeats: number of local and global views to use in the dino loss
        :return: loss values
        """
        loss = torch.tensor(0., device=self.device)
        for dataset in ['dino', 'ibot']:
            if dataset == 'ibot':
                batch, masks = self.train_loader.create_ibot_batch(batch_size=batch_size,
                                                                   crop_size=crop_size)
                batch, masks = batch.to(self.device), masks.to(self.device).type(torch.float32)
                kernel = torch.ones((1, 3, self.config['model']['patch_size'],
                                     self.config['model']['patch_size']),
                                    dtype=torch.float32).to(batch.device)
                masks_encoded = torch.nn.functional.conv2d(masks,
                                                           kernel,
                                                           stride=self.config['model']['patch_size']).flatten(
                    2).transpose(1, 2)
                masks_encoded = (masks_encoded == 0).to(self.device)
                masks_encoded = torch.cat(
                    (torch.zeros((len(masks_encoded), 1, 1), dtype=torch.bool).to(masks.device), masks_encoded),
                    dim=1)
                features_student = self.student(batch,
                                                mask_token=self.student.embeddings.mask_token.unsqueeze(1),
                                                ibot_mask=masks_encoded)
                with torch.no_grad():
                    features_teacher = self.teacher(batch)
                s = self.head_s(features_student)
                t = self.head_t(features_teacher)

                loss_ibot = self.ibot_loss(s, t, self.student_temp, self.teacher_temp, ibot_masks=masks_encoded)
                loss += self.ibot_coef * loss_ibot
                del batch, s, t, features_student, features_teacher
            else:
                s_batch, t_batch = self.train_loader.create_dino_batch(batch_size=batch_size,
                                                                       repeats=repeats,
                                                                       crop_size=(
                                                                           (self.config['training']['local_size'],
                                                                            self.config['training']['local_size']),
                                                                           crop_size))
                s_batch, t_batch = s_batch.to(self.device), t_batch.to(self.device)
                features_student = torch.stack(self.student(s_batch.flatten(0, 1)).chunk(batch_size), dim=0)
                with torch.no_grad():
                    features_teacher = self.teacher(t_batch.flatten(0, 1))

                features_teacher = torch.stack(features_teacher.chunk(batch_size), dim=0)
                loss_koleo_p = self.koleo_loss(features_student[:, :, 1:].flatten(2).flatten(0, 1),
                                               self.memory_bank[:, 1:].flatten(1))
                loss += self.koleo_coef * loss_koleo_p
                loss_koleo_c = self.koleo_loss(features_student[:, 0, 0], self.memory_bank[:, 0])
                loss += self.koleo_coef * loss_koleo_c
                for _ in range(len(features_student)):
                    __ = np.random.choice(len(self.memory_bank))
                    self.memory_bank[__] = features_student[_][0].clone().detach()

                features_student = self.get_class_token(features_student,
                                                        (self.config['training']['local_size'],
                                                         self.config['training']['local_size']),
                                                        self.config['model']['patch_size'],
                                                        self.config['model']['hidden_size'])
                features_teacher = self.get_class_token(features_teacher,
                                                        crop_size,
                                                        self.config['model']['patch_size'],
                                                        self.config['model']['hidden_size'])

                s = self.head_s(features_student)
                t = self.head_t(features_teacher)
                loss_dino = self.dino_loss(s, t, self.student_temp, self.teacher_temp)
                loss += self.dino_coef * loss_dino

                del s_batch, t_batch, features_student, features_teacher
            torch.cuda.empty_cache()
        return loss, loss_dino.clone().detach().item(), loss_ibot.clone().detach().item(), loss_koleo_c.clone().detach().item(), loss_koleo_p.clone().detach().item()

    def training_cycle(self,
                       stage: int,
                       epochs: int,
                       steps_per_epoch: int,
                       steps_per_epoch_val: int,
                       tps_0: float,
                       tpt_0: float,
                       tps_f: float,
                       tpt_f: float,
                       batch_size: int,
                       repeats: tuple):
        """
        :param stage: stage of training. 0 for warmup, 1 for regular training
        :param epochs: max number of epochs to do
        :param steps_per_epoch: amount of step on each epoch
        :param steps_per_epoch_val: amount of batches on which to validate
        :param tps_0: initial temperature for student
        :param tpt_0:  initial temperature for teacher
        :param tps_f: final temperature for student
        :param tpt_f: final temperature for teacher
        :param batch_size: batch size
        :param repeats: amount of local and global views for dino loss
        :return: validation loss
        """
        self.dino_coef = torch.tensor(self.config['training']['dino_coef'], device=self.device)
        self.ibot_coef = torch.tensor(self.config['training']['ibot_coef'], device=self.device)
        self.koleo_coef = torch.tensor(self.config['training']['koleo_coef'], device=self.device)
        early_stop = 0
        for epoch in range(epochs):
            self.create_memory_bank(loader=self.train_loader,
                                    crop_size=(self.config['training']['crop_size'],
                                               self.config['training']['crop_size']))
            tps = tps_f + (tps_0 - tpt_f) * np.cos(np.pi * epoch / (2 * epochs))
            tpt = tpt_f + (tpt_0 - tpt_f) * np.cos(np.pi * epoch / (2 * epochs))
            self.student_temp = tps
            self.teacher_temp = tpt
            self.train()
            progress_bar = trange(steps_per_epoch, desc='Beginning epoch', leave=True)
            mean_train_loss, mean_dino_loss, mean_ibot_loss = torch.tensor(0.), torch.tensor(0.), torch.tensor(0.)
            mean_koleo_c_loss = torch.tensor(0.)
            mean_koleo_p_loss = torch.tensor(0.)
            for step in progress_bar:
                optimizer = self.optimizer
                optimizer.zero_grad()
                loss, loss_dino, loss_ibot, loss_koleo_c, loss_koleo_p = self.training_step(batch_size=batch_size,
                                                                                            crop_size=(
                                                                                                self.config['training'][
                                                                                                    'crop_size'],
                                                                                                self.config['training'][
                                                                                                    'crop_size']),
                                                                                            repeats=repeats)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.student.parameters(), 3)
                optimizer.step()
                self.update_teacher()
                if loss.item() == torch.nan:
                    break
                mean_train_loss += loss.detach().item()
                mean_dino_loss += loss_dino
                mean_ibot_loss += loss_ibot
                mean_koleo_c_loss += loss_koleo_c
                mean_koleo_p_loss += loss_koleo_p

                progress_bar.set_description(
                    f'Epoch: {epoch} Error: {int(1000 * mean_train_loss.item() / (step + 1)) / 1000}, dino: {int(1000 * mean_dino_loss / (step + 1)) / 1000}, ibot: {int(1000 * mean_ibot_loss / (step + 1)) / 1000}, koleo_c: {int(1000 * mean_koleo_c_loss / (step + 1)) / 1000}, koleo_p: {int(1000 * mean_koleo_p_loss / (step + 1)) / 1000}, Best: {self.best}')
                progress_bar.refresh()

                torch.cuda.empty_cache()
                del loss, loss_dino, loss_ibot, loss_koleo_c, loss_koleo_p

            self.history[0].append(mean_train_loss / steps_per_epoch)
            self.create_memory_bank(loader=self.val_loader,
                                    crop_size=(self.config['training']['crop_size'],
                                               self.config['training']['crop_size']))
            scheduler = self.scheduler
            self.eval()
            with torch.no_grad():
                val_loss = torch.tensor(0.).to(self.device)
                for step in range(steps_per_epoch_val):
                    val_loss_step, loss_dino, loss_ibot, loss_koleo_c, loss_koleo_p = self.training_step(
                        batch_size=batch_size,
                        crop_size=(self.config['training']['crop_size'],
                                   self.config['training']['crop_size']),
                        repeats=repeats)
                    val_loss += val_loss_step.to(self.device)
                    torch.cuda.empty_cache()
                val_loss = val_loss / steps_per_epoch_val
                self.history[1].append(val_loss.detach().item())
            scheduler.step(val_loss)
            early_stop = self.save_models(val_loss,
                                          early_stop,
                                          stage,
                                          thr=self.config['checkpoint']['saving_thr'])

            if early_stop > self.config['training']['early_stop']:
                break

            torch.cuda.empty_cache()
            del val_loss

    def training_stage(self,
                       lr: float,
                       stage: int,
                       epochs: int,
                       steps_per_epoch: int,
                       steps_val: int,
                       tps_0: float,
                       tpt_0: float,
                       tps_f: float,
                       tpt_f: float,
                       batch_size: int,
                       repeats: tuple):
        """
        :param lr: learning rate
        :param stage: training stage. 0 for warmup, 1 for regular training
        :param epochs: max number of epochs
        :param steps_per_epoch: steps per epoch
        :param steps_val: amount of batches to validate on
        :param tps_0: initial temperature for student
        :param tpt_0: initial temperature for teacher
        :param tps_f: final temperature for student
        :param tpt_f: final temperature for teacher
        :param batch_size: batch size
        :param repeats: number of local and global views for dino loss
        :return: None. Best and last model's weights are saved
        """
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(0.9, 0.999))
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min',
                                                                    patience=self.config['training']['patience'],
                                                                    factor=self.config['training']['factor'],
                                                                    min_lr=self.config['training']['min_lr'])
        self.teacher.load_state_dict(self.student.state_dict())
        self.training_cycle(stage, epochs, steps_per_epoch, steps_val, tps_0,
                            tpt_0, tps_f, tpt_f, batch_size, repeats)

    def do_train(self):
        self.best = torch.inf
        epochs_per_stage = [self.config['training']['epochs_warmup'],
                            self.config['training']['epochs']]
        lr_per_stage = [float(self.config['training']['lr']),
                        float(self.config['training']['lr'])]
        # number of local and global crops
        repeats_list = [(4, 2), (4, 2)]
        for stage in np.arange(2):
            self.best = torch.inf
            print(f'Stage {stage}')
            if not os.path.exists(
                    f'{self.config["checkpoint"]["save_path"]}/{self.config["checkpoint"]["name"]}/student_{stage}_last.pt'):
                if os.path.exists(
                        f'{self.config["checkpoint"]["save_path"]}/{self.config["checkpoint"]["name"]}/student_{stage}_best.pt'):
                    self.student.load_state_dict(torch.load(
                        f'{self.config["checkpoint"]["save_path"]}/{self.config["checkpoint"]["name"]}/student_{stage}_best.pt'))
                    self.teacher.load_state_dict(torch.load(
                        f'{self.config["checkpoint"]["save_path"]}/{self.config["checkpoint"]["name"]}/student_{stage}_best.pt'))
                self.training_stage(lr_per_stage[stage], stage, epochs_per_stage[stage],
                                    steps_per_epoch=self.config['training']['steps_per_epoch'],
                                    steps_val=self.config['training']['steps_per_epoch_val'],
                                    tps_0=self.config['training']['tps_0'],
                                    tpt_0=self.config['training']['tpt_0'],
                                    tps_f=self.config['training']['tps_f'],
                                    tpt_f=self.config['training']['tpt_f'],
                                    batch_size=self.config['training']['batch_size'],
                                    repeats=repeats_list[stage])
                torch.save(self.student.state_dict(),
                           f'{self.config["checkpoint"]["save_path"]}/{self.config["checkpoint"]["name"]}/student_{stage}_last.pt')
            else:
                self.student.load_state_dict(
                    torch.load(
                        f'{self.config["checkpoint"]["save_path"]}/{self.config["checkpoint"]["name"]}/student_{stage}_best.pt',
                        weights_only=True))
                self.teacher.load_state_dict(
                    torch.load(
                        f'{self.config["checkpoint"]["save_path"]}/{self.config["checkpoint"]["name"]}/student_{stage}_best.pt',
                        weights_only=True))

        np.save(f'{self.config["checkpoint"]["save_path"]}/{self.config["checkpoint"]["name"]}/history.npy',
                np.array(self.history))


def main(data_path, config_path):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = TrainingModel(config_path=config_path,
                          train_loader=Loader(data_path=data_path, train_loader=True),
                          val_loader=Loader(data_path=data_path, train_loader=False),
                          device=device).to(device)

    checkpoint_path = f'{model.config["checkpoint"]["save_path"]}/{model.config["checkpoint"]["name"]}'
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    # Start training
    print('Beginning training', model.config["checkpoint"]["name"])
    model.do_train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with specified data and configuration paths.")
    parser.add_argument('data_path', type=str, help="Path to the dataset")
    parser.add_argument('config_path', type=str, help="Path to the configuration YAML file")

    args = parser.parse_args()

    main(args.data_path, args.config_path)
