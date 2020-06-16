import os
import torch
import pandas as pd
from torch.utils.data import Dataset

from utils.data_utils import tile
from utils.minimal_image import get_minimal_image

from skimage.io import MultiImage

import cv2 as cv
import numpy as np


class TrainDataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_dir: str, transforms=None, outputs=6):
        self.images = [i + ".png" for i in df["image_id"]]
        self.labels = [i for i in df["isup_grade"]]
        self.data_dir = data_dir
        self.transforms = transforms
        self.outputs = outputs

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        name = self.images[item]
        path = os.path.join(self.data_dir, name)
        label = self.labels[item]

        img = cv.imread(path)
        if self.transforms is not None:
            img = self.transforms(image=img)["image"]

        img = torch.from_numpy(img.transpose(2, 0, 1))
        return img, label


class TrainDatasetBinning(Dataset):
    def __init__(self, df: pd.DataFrame, data_dir: str, transforms=None, tiff_scale=1, sz=128, N=100, random=True):
        self.images = [i + ".tiff" for i in df["image_id"]]
        self.labels = [i for i in df["isup_grade"]]
        self.data_dir = data_dir
        self.transforms = transforms
        self.tiff_scale = tiff_scale
        self.sz = sz
        self.N = N
        self.random = random

    def __len__(self):
        return len(self.images)

    def _process_by_tiles(self, img):
        s = int(np.sqrt(self.N))
        result = np.zeros([self.sz * s, self.sz * s, 3], dtype=np.uint8)

        indexes = np.arange(self.N)
        if self.random:
            np.random.shuffle(indexes)

        for i, j in enumerate(indexes):
            x = j % s
            y = j // s

            xo = i % s
            yo = i // s

            img_i = img[yo * self.sz:(yo + 1) * self.sz, xo * self.sz:(xo + 1) * self.sz]
            if self.transforms is not None:
                img_i = self.transforms(image=img_i)["image"]

            if result.dtype != img_i.dtype:
                result = result.astype(img_i.dtype)
            result[y * self.sz:(y + 1) * self.sz, x * self.sz:(x + 1) * self.sz] = img_i

        return result

    def __getitem__(self, item):
        name = self.images[item]
        path = os.path.join(self.data_dir, name)
        label = self.labels[item]

        label = np.array([(1 if i < label else 0) for i in range(5)], dtype=np.float32)

        img = MultiImage(path)[self.tiff_scale]
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

        img = tile(img, self.sz, self.N, self.transforms, self.random)

        img = torch.from_numpy(img.transpose(2, 0, 1))
        return img, label


class TrainDatasetBinningTiled(TrainDatasetBinning):
    def __getitem__(self, item):
        name = self.images[item]
        path = os.path.join(self.data_dir, name)
        label = self.labels[item]

        label = np.array([(1 if i < label else 0) for i in range(5)], dtype=np.float32)

        img = cv.imread(path.replace(".tiff", ".png"))

        img = self._process_by_tiles(img)

        img = torch.from_numpy(img.transpose(2, 0, 1))
        return img, label


class InferDataset(Dataset):
    def __init__(self, images: list, data_dir: str, tile_size: int, num_tiles: int, tiff_scale=-1, transforms=None):
        self.images = images
        self.data_dir = data_dir
        self.transforms = transforms
        self.tile_size = tile_size
        self.num_tiles = num_tiles
        self.tiff_scale = tiff_scale

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        name = self.images[item]
        if not os.path.splitext(name)[1]:
            name += ".tiff"
        path = os.path.join(self.data_dir, name)

        img = MultiImage(path)[self.tiff_scale]
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        img, _ = get_minimal_image(img)
        img = tile(img, self.tile_size, self.num_tiles)
        if self.transforms is not None:
            img = self.transforms(image=img)["image"]

        img = torch.from_numpy(img.transpose(2, 0, 1))
        return img


if __name__ == "__main__":
    import json
    import matplotlib.pyplot as plt
    import albumentations as A
    from tqdm import tqdm

    df = pd.read_csv("../input/prostate-cancer-grade-assessment/train.csv")
    # with open("../input/compact_representation.json", "r") as file:
    #     compact_representation = json.load(file)

    mean = [127.66098, 127.66102, 127.66085]
    std = [10.5911, 10.5911045, 10.591107]

    transforms = A.Compose(
            [
                A.InvertImg(p=1),
                A.RandomSizedCrop([100, 128], 128, 128),
                A.Transpose(),
                A.Flip(),
                A.Rotate(90, border_mode=cv.BORDER_CONSTANT, value=(0, 0, 0)),
                A.RandomBrightnessContrast(0.02, 0.02),
                A.HueSaturationValue(0, 10, 10),
                # A.Normalize(mean, std, 1),
            ]
        )

    # dataset = TrainDataset(df, "/datasets/panda/train_64_100", transforms)
    dataset = TrainDatasetBinningTiled(pd.read_csv("../input/prostate-cancer-grade-assessment/train.csv"),
                                     "/datasets/panda/train_128_100/",
                                     transforms,
                                     1, 128, 100)

    # x_tot, x2_tot = [], []
    # for img, _ in tqdm(dataset):
    #     img = img.cpu().numpy().astype(np.float32)
    #     x_tot.append(img.reshape(-1, 3).mean(0))
    #     x2_tot.append((img ** 2).reshape(-1, 3).mean(0))
    # img_avr = np.array(x_tot).mean(0)
    # img_std = np.sqrt(np.array(x2_tot).mean(0) - img_avr ** 2)
    # print('mean:', img_avr, ', std:', np.sqrt(img_std))


    img, label = dataset[10000]
    img = img.numpy().transpose(1, 2, 0)
    print(img.min(), img.max())
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()
