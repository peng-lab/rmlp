import cv2
import torch
from transformers import AutoImageProcessor
from glob import glob
import numpy as np

class Loader:
    def __init__(self, data_path: str):
        """
        :param data_path: path to the NYU-Depth V2 dataset
        """
        self.data_path = data_path
        folders = np.array(glob(f'{data_path}/data/nyu2_train/*'))
        images_train, images_val = [], []
        for folder in folders:
            images = np.sort(glob(f'{folder}/*.jpg'))
            images_train.append(images[:int(0.8*len(images))])
            images_val.append(images[int(0.8*len(images)):])

        self.images_train = np.concatenate(images_train)
        self.images_val = np.concatenate(images_val)

        self.image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")


    def __call__(self, batch_size: int, db: str,crop_size: tuple = (224,224)):
        """
        :param batch_size: batch size
        :param db: "train" or "val" for training or validation batches
        :param crop_size: size of output images
        :return: image and depth estimation batches a tensors
        """
        batch_im,batch_an = [],[]
        for _ in range(batch_size):

            chosen = np.random.choice(len(getattr(self,f'images_{db}')))
            chosen = getattr(self,f'images_{db}')[chosen]
            im = cv2.imread(chosen)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            an = cv2.imread(f'{self.data_path}/data/nyu2_train/{chosen.split("/")[-2]}/{chosen.split("/")[-1].split(".")[0]}.png')

            im = cv2.resize(im, crop_size, interpolation=cv2.INTER_LINEAR)
            an = cv2.resize(an, crop_size, interpolation=cv2.INTER_LINEAR)[...,0]

            an = torch.tensor(an,dtype=torch.float32)

            batch_im.append(im)
            batch_an.append(an)

        batch_im = self.image_processor(batch_im, return_tensors="pt")['pixel_values']
        batch_an = torch.stack(batch_an)
        batch_an = batch_an[:, 14:-14, 14:-14]
        batch_an = torch.nn.functional.interpolate(
            batch_an.unsqueeze(1).type(torch.float32),
            size=crop_size,
            mode="nearest-exact")[:, 0]
        batch_an -= batch_an.flatten(1).amin(dim=-1).view((-1,1,1))
        batch_an /= batch_an.flatten(1).amax(dim=-1).view((-1, 1, 1))
        batch_an *= 49

        return batch_im,batch_an.type(torch.long)

class TestLoader:
    def __init__(self,data_path: str):
        """
        :param data_path: path to the NYU-Depth V2 dataset
        """
        self.data_path = data_path
        self.names = np.array(glob(f'{data_path}/data/nyu2_test/*_colors.png'))
        self.image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

    def get_from_name(self, name: str, crop_size: tuple = (224,224)):
        """
        :param name: name of a test image to load and preprocess
        :param crop_size: crop size of image and depth estimation
        :return: image and ground truth as tensors
        """
        image = cv2.imread(name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, crop_size, interpolation=cv2.INTER_LINEAR)
        image = self.image_processor(image, return_tensors="pt")['pixel_values']

        label = cv2.imread(f'{name.split("_colors")[-2]}_depth.png')
        label = cv2.resize(label, crop_size,
               interpolation=cv2.INTER_LINEAR)[...,0]
        label = torch.tensor(label,dtype=torch.float32)
        label = label[14:-14, 14:-14].unsqueeze(0)
        label = torch.nn.functional.interpolate(
            label.unsqueeze(1).type(torch.float32),
            size=crop_size,
            mode="nearest-exact")[:, 0]
        label -= label.flatten(1).amin(dim=-1).view((-1,1,1))
        label = 49 * label / label.flatten(1).amax(dim=-1).view((-1,1,1))

        return image, label.type(torch.long)
