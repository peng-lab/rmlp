from glob import glob
import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor


class Loader:
    def __init__(self,
                 data_path: str,
                 train_loader: bool):
        """
        :param data_path: path were ImageNet1K is stored
        :param train_loader: True if it's for training, False if for validation
        """
        self.datasets = ['imagenet1k']
        self.train_loader = train_loader
        self.data_path = data_path
        self.image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

    def get_path(self,
                 dataset: str):
        """
        :param dataset: dataset from where to read the image
        :return: an image path
        """
        folder = np.random.choice(glob(f'{self.data_path}/{dataset}/*'))
        samples = glob(f'{folder}/*.jpg')
        if self.train_loader:
            return np.random.choice(samples[int(0.85 * len(samples)):])
        else:
            return np.random.choice(samples[:int(0.85 * len(samples))])

    def data_augmentation(self,
                          x: np.array,
                          crop_size: tuple):
        """
        :param x: image
        :param crop_size: size of the crop
        :return: augmented version of the image
        """

        if np.random.rand() < 0.5:
            x = cv2.flip(x, np.random.randint(-1, 2))

        if np.random.rand() < 0.5:
            cx, cy = crop_size  # center of rotation
            rand_angle = np.random.randint(-180, 180)  # random angle range
            m = cv2.getRotationMatrix2D((cy // 2, cx // 2), rand_angle, 1)  # center angle scale
            x = cv2.warpAffine(x, m, crop_size, borderMode=cv2.BORDER_REFLECT)

        if np.random.rand() < 0.5:
            x = cv2.cvtColor(x, cv2.COLOR_RGB2HSV)  # transform to HSV color space .
            h, s, v = cv2.split(x)  # split each channel in order to add seperate range of values to each channel.
            h += np.random.randint(0, 3, size=crop_size, dtype=np.uint8)
            s += np.random.randint(0, 3, size=crop_size, dtype=np.uint8)
            v += np.random.randint(0, 1, size=crop_size, dtype=np.uint8)
            x = cv2.merge([h, s, v])
            x = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)

        if np.random.rand() < 0.5:
            blur_val = np.random.randint(3, 8)  # blur value random
            x = cv2.blur(x, (blur_val, blur_val))

        return x

    def create_dino_batch(self,
                          batch_size: int,
                          repeats: tuple = (10, 2),
                          crop_size: tuple = ((96, 96), (224, 224))):
        """
        :param batch_size: batch size
        :param repeats: number of local and glocal views for dino loss
        :param crop_size: size of crop
        :return: preprocessed batches as tensors for student and teacher during training with the dino loss
        """
        s_batch, t_batch = [], []
        for _ in range(batch_size):
            while True:
                image = cv2.imread(self.get_path(np.random.choice(self.datasets)))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if (image.shape[0] > crop_size[1][0]) and (image.shape[1] > crop_size[1][1]):
                    break

            image = cv2.resize(image, (int(1.1 * crop_size[1][0]), int(1.1 * crop_size[1][0])),
                               interpolation=cv2.INTER_LINEAR)
            s_batch.append([])
            t_batch.append([])

            for r in range(repeats[0]):
                i = int(np.random.uniform(image.shape[0] - crop_size[0][0]))
                j = int(np.random.uniform(image.shape[1] - crop_size[0][1]))
                crop = image[i:i + crop_size[0][0],
                             j:j + crop_size[0][1]]
                crop = self.data_augmentation(crop, crop_size=crop_size[0])
                s_batch[-1].append(self.image_processor(crop, return_tensors="pt")['pixel_values'])
                s_batch[-1][-1] = torch.nn.functional.interpolate(s_batch[-1][-1],
                                                                  size=crop_size[0],
                                                                  mode="bilinear",
                                                                  align_corners=False,
                                                                  )[0]
            for r in range(repeats[1]):
                i = int(np.random.uniform(image.shape[0] - crop_size[1][0]))
                j = int(np.random.uniform(image.shape[1] - crop_size[1][1]))
                crop = image[i:i + crop_size[1][0],
                             j:j + crop_size[1][1]]
                crop = self.data_augmentation(crop, crop_size=crop_size[1])
                t_batch[-1].append(self.image_processor(crop, return_tensors="pt")['pixel_values'])
                t_batch[-1][-1] = torch.nn.functional.interpolate(t_batch[-1][-1],
                                                                  size=crop_size[1],
                                                                  mode="bilinear",
                                                                  align_corners=False,
                                                                  )[0]
            s_batch[-1], t_batch[-1] = torch.stack(s_batch[-1]), torch.stack(t_batch[-1])
        s_batch, t_batch = torch.stack(s_batch), torch.stack(t_batch)
        return s_batch, t_batch

    def create_ibot_batch(self,
                          batch_size: int,
                          crop_size: tuple = (224, 224),
                          patch_size: int = 14):
        """
        :param batch_size: batch size
        :param crop_size: size of crop
        :param patch_size: size of patched to mask
        :return: batch of preprocessed images and corresponding masks as tensors
        """
        batch, masks = [], []
        for _ in range(batch_size):
            while True:
                image = cv2.imread(self.get_path(np.random.choice(self.datasets)))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if (image.shape[0] > crop_size[0]) and (image.shape[1] > crop_size[1]):
                    break

            image = cv2.resize(image, (int(1.1 * crop_size[0]), int(1.1 * crop_size[0])),
                               interpolation=cv2.INTER_LINEAR)
            i = int(np.random.normal(image.shape[0] // 2, image.shape[0] // 5))
            j = int(np.random.normal(image.shape[1] // 2, image.shape[1] // 5))

            i = max(min(i, image.shape[0] - crop_size[0] // 2), crop_size[0] // 2)
            j = max(min(j, image.shape[1] - crop_size[1] // 2), crop_size[1] // 2)
            crop = image[i - crop_size[0] // 2:i + crop_size[0] // 2,
                         j - crop_size[1] // 2:j + crop_size[1] // 2]
            crop = self.data_augmentation(crop, crop_size=crop_size)
            crop = self.image_processor(crop, return_tensors="pt")['pixel_values'][0].permute((1, 2, 0))

            kernel = torch.ones(((crop_size[0] // patch_size) * (crop_size[1] // patch_size), patch_size, patch_size))
            kernel *= torch.tensor(np.random.randint(0, 2,
                                                     size=(
                                                         (crop_size[0] // patch_size) * (crop_size[1] // patch_size), 1,
                                                         1)))
            kernel *= torch.tensor(np.random.randint(0, 2,
                                                     size=(
                                                         (crop_size[0] // patch_size) * (crop_size[1] // patch_size), 1,
                                                         1)))

            kernel = torch.cat(kernel.chunk((crop_size[0] // patch_size), dim=0), dim=-1).flatten(0, 1)
            kernel = kernel.unsqueeze(-1).repeat((1, 1, 3)).type(crop.dtype)
            crop = torch.nn.functional.interpolate(crop.clone().permute(2, 0, 1).unsqueeze(0),
                                                   size=crop_size,
                                                   mode="bilinear",
                                                   align_corners=False,
                                                   )[0]
            batch.append(crop)
            masks.append(kernel.clone().permute(2, 0, 1))

        batch, masks = torch.stack(batch).type(torch.float32), torch.stack(masks)
        return batch, masks

    def retrieve_images(self,
                        batch_size: int,
                        crop_size: tuple):
        """
        :param batch_size: batch size
        :param crop_size: size of crops
        :return: patch of preprocessed images as tensors
        """
        batch = []
        for _ in range(batch_size):
            image = cv2.imread(self.get_path(np.random.choice(self.datasets)))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, crop_size,
                               interpolation=cv2.INTER_LINEAR)
            batch.append(torch.tensor(image))
        batch = torch.stack(batch)
        batch = self.image_processor(batch, return_tensors="pt")['pixel_values']
        return batch
