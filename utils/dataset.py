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

    df = pd.read_csv("/home/dipet/kaggle/prostate/input/prostate-cancer-grade-assessment/train.csv")
    with open("../input/compact_representation.json", "r") as file:
        compact_representation = json.load(file)

    transforms = A.Compose(
        [
            A.Compose(
                [
                    A.OneOf([A.GaussNoise(), A.MultiplicativeNoise(elementwise=True)]),
                    A.RandomBrightnessContrast(0.02, 0.02),
                    A.HueSaturationValue(0, 10, 10),
                    A.Flip(),
                    A.RandomGridShuffle(grid=(10, 10)),
                    A.GridDistortion(),
                ],
                p=0.5,
            ),
            A.ToFloat(),
        ]
    )

    dataset = TrainDataset(df, "/datasets/panda/train_64_100", transforms)
    img, label = dataset[0]
    img = img.numpy().transpose(1, 2, 0)
    print(img.min(), img.max())
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()
