import os
import numpy as np
import torch
from torch import nn
from tqdm import trange,tqdm
from typing import Callable


class DST(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 head: nn.Module,
                 loss_function: Callable,
                 test_metrics: dict,
                 loader: Callable,
                 test_loader: Callable,
                 save_path: str = '.',
                 name: str = 'taks_1',
                 unet: bool = True):
        """
        :param encoder: model containing the ViT
        :param head: downstream head
        :param loss_function: loss function for task
        :param test_metrics: dictionry with test metrics
        :param loader: loader of images and ground truth for training and validation
        :param test_loader: loader for images and groun truth for testing
        :param save_path: path where to save the downstream model
        :param name: name for the model
        :param unet: True if downstream task is to be learned with a ViT-UNet hybrid
        """
        super().__init__()
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.unet = unet

        self.encoder = encoder
        self.encoder.to(self.device)
        if unet:
            for param in self.encoder.encoder.parameters():
                param.requires_grad = False
        else:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.head = head
        self.head.to(self.device)

        self.save_path = save_path
        self.loss_function = loss_function
        self.history = [[], []]
        self.name = name
        self.loader = loader
        self.test_loader = test_loader
        self.test_metrics = test_metrics

    def forward(self,
                x: torch.Tensor):
        if self.unet:
            return self.encoder(x)
        else:
            with torch.no_grad():
                embeddings = self.encoder(x)
            if 'dinov2' in self.name:
                embeddings = embeddings.last_hidden_state
            return self.head(embeddings)

    def training_cycle(self,
                       batch_size: int,
                       epoch: int,
                       steps_per_epoch: int,
                       steps_per_val: int):
        """
        :param batch_size: batch size
        :param epoch: current epoch
        :param steps_per_epoch: steps per training epoch
        :param steps_per_val: number of batched to validate on
        :return: validation loss
        """
        self.train()
        mean_train_loss = torch.tensor(0.)
        if steps_per_epoch is not None:
            progress_bar = trange(steps_per_epoch, desc='Beginning epoch', leave=True)
            train_mean_denominator = steps_per_epoch
        else:
            progress_bar = trange(len(self.loader.train_tomograms)//batch_size, desc='Beginning epoch', leave=True)
            train_mean_denominator = len(self.loader.train_tomograms)//batch_size

        for n_batch in progress_bar:
            inputs,target = self.loader(batch_size,'train')
            inputs, target = inputs.to(self.device), target.to(self.device)
            prediction = self(inputs)
            loss = self.loss_function(prediction, target)
            optimizer = self.optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mean_train_loss += loss.detach().item()

            progress_bar.set_description(
                f'Epoch: {epoch} Error: {mean_train_loss.item() / (n_batch+1)}, Best: {self.best}')
            progress_bar.refresh()
            torch.cuda.empty_cache()

        self.history[0].append(mean_train_loss/train_mean_denominator)

        self.eval()
        val_loss = torch.tensor(0.)
        scheduler = self.scheduler
        if steps_per_val is not None:
            range_size = steps_per_val
        else:
            range_size = len(self.val_loader.val_tomograms)//batch_size
        for n_batch in range(range_size):
            inputs, target = self.loader(batch_size,'val')
            inputs, target = inputs.to(self.device), target.to(self.device)
            with torch.no_grad():
                prediction = self(inputs)
            val_loss += self.loss_function(prediction,target).detach().item()
        val_loss = val_loss/(range_size)
        self.history[1].append(val_loss)
        scheduler.step(val_loss)

        if self.best > val_loss:
            self.best = val_loss
            torch.save(self.head.state_dict(),f'{self.save_path}/{self.name}_head_best.pt')
            torch.save(self.encoder.state_dict(), f'{self.save_path}/{self.name}_encoder_best.pt')

        return val_loss

    def do_train(self,
                 epochs: int,
                 steps_per_epoch: int,
                 steps_per_val: int,
                 batch_size: int,
                 lr=2.0e-3):
        """
        :param epochs: max number of epochs to train for
        :param steps_per_epoch: steps per training epoch
        :param steps_per_val: number of batches to validate on
        :param batch_size: batch size
        :param lr: learning rate
        :return: None
        """
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        if not os.path.exists(f'{self.save_path}/{self.name}_head_last.pt'):
            self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(0.9, 0.999))
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=3,
                                                                        factor=0.4,
                                                                        min_lr=1.e-7)
            self.best = torch.inf

            early_stop = 0
            for epoch in range(epochs):
                val_loss = self.training_cycle(batch_size=batch_size,
                                               epoch=epoch,
                                               steps_per_epoch=steps_per_epoch,
                                               steps_per_val=steps_per_val)

                if self.best == val_loss:
                    early_stop = 0
                else:
                    early_stop += 1
                if early_stop == 10:
                    break

            torch.save(self.head.state_dict(), f'{self.save_path}/{self.name}_head_last.pt')
            torch.save(self.encoder.state_dict(), f'{self.save_path}/{self.name}_encoder_last.pt')
            np.save(f'{self.save_path}/{self.name}_history.npy',np.array(self.history))
            self.head.load_state_dict(torch.load(f'{self.save_path}/{self.name}_head_best.pt'))
            self.encoder.load_state_dict(torch.load(f'{self.save_path}/{self.name}_encoder_best.pt'))
        else:
            self.head.load_state_dict(torch.load(f'{self.save_path}/{self.name}_head_best.pt'))
            self.encoder.load_state_dict(torch.load(f'{self.save_path}/{self.name}_encoder_best.pt'))

    @torch.no_grad()
    def do_evaluate(self,
                    path_test_results: str,
                    task='classification'):
        """
        :param path_test_results: path to save the test results
        :param task: name of the task to train on
        :return: None
        """
        self.eval()
        if not os.path.exists(f'{path_test_results}/test_samples'):
            os.mkdir(f'{path_test_results}/test_samples')
        if (not os.path.exists(f'{path_test_results}/test_samples/{self.name}')) and (int(self.name[-1])<4):
            os.mkdir(f'{path_test_results}/test_samples/{self.name}')

        if task == 'classification':
            metric_results = {}
            predictions,targets = [],[]
            for _,name in enumerate(tqdm(self.test_loader.names)):
                image, target = self.test_loader.get_from_name(name)
                image,target = image.to(self.device),target.to(self.device)[0]
                prediction = self(image)[0]

                predictions.append(torch.argmax(prediction,dim=-1))
                targets.append(target)

            predictions,targets = torch.tensor(predictions),torch.tensor(targets)
            for metric_name, metric in self.test_metrics.items():
                metric_results[metric_name] = metric(predictions, targets).item()

            np.save(f'{path_test_results}/{self.name}.npy', metric_results)

        elif task == 'slice_segmentation':
            metric_results = {metric_name:[] for metric_name in self.test_metrics.keys()}
            for _, name in enumerate(tqdm(self.test_loader.names)):
                image, target = self.test_loader.get_from_name(name)
                image, target = image.to(self.device), target.to(self.device)
                prediction = torch.stack([self(i.unsqueeze(0))[0] for i in image])
                for metric_name, metric in self.test_metrics.items():
                    metric_results[metric_name].append(torch.tensor([metric(prediction[_].unsqueeze(0),
                                                                             target[_].unsqueeze(0))
                                                                      for _ in range(len(prediction))]))
                torch.cuda.empty_cache()
            for metric_name, metric in self.test_metrics.items():
                metric_results[metric_name] = torch.cat(metric_results[metric_name]).mean().item()
            np.save(f'{path_test_results}/{self.name}.npy', metric_results)

        else:
            metric_results = {}
            for _,name in enumerate(tqdm(self.test_loader.names)):
                image, target = self.test_loader.get_from_name(name)
                image,target = image.to(self.device),target.to(self.device)
                prediction = self(image)
                metric_results[name.item()] = {}
                for metric_name,metric in self.test_metrics.items():
                    metric_results[name][metric_name] = metric(prediction,target).item()
                torch.cuda.empty_cache()

            np.save(f'{path_test_results}/{self.name}.npy', metric_results)
