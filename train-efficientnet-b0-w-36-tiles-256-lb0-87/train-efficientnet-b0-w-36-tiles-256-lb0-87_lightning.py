DEBUG = False

import os
import time
import skimage.io
import numpy as np
import pandas as pd
import cv2
import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler, SequentialSampler
from warmup_scheduler import GradualWarmupScheduler
from efficientnet_pytorch import model as enet
import albumentations
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm_notebook as tqdm
from pytorch_lightning.core.lightning import LightningModule
import random
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

SAVE_NAME = "effnetb0_256_36_lb087"
FP16 = True
batch_size = 2
num_workers = min(batch_size, 8)
ACCUM_STEPS = 1

data_dir = '../input/prostate-cancer-grade-assessment'
df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
image_folder = os.path.join(data_dir, 'train_images')

kernel_type = 'how_to_train_effnet_b0_to_get_LB_0.86'

enet_type = 'efficientnet-b0'
fold = 0
tile_size = 256
image_size = 256
n_tiles = 36
out_dim = 5
init_lr = 3e-4
warmup_factor = 10

warmup_epo = 1
n_epochs = 1 if DEBUG else 30
df_train = df_train.sample(100).reset_index(drop=True) if DEBUG else df_train

device = torch.device('cuda')

transforms_train = albumentations.Compose([
    albumentations.Transpose(p=0.5),
    albumentations.VerticalFlip(p=0.5),
    albumentations.HorizontalFlip(p=0.5),
])
transforms_val = albumentations.Compose([])

print(image_folder)

skf = StratifiedKFold(5, shuffle=True, random_state=42)
df_train['fold'] = -1
for i, (train_idx, valid_idx) in enumerate(skf.split(df_train, df_train['isup_grade'])):
    df_train.loc[valid_idx, 'fold'] = i


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(0)


def get_tiles(img, mode=0):
    result = []
    h, w, c = img.shape
    pad_h = (tile_size - h % tile_size) % tile_size + ((tile_size * mode) // 2)
    pad_w = (tile_size - w % tile_size) % tile_size + ((tile_size * mode) // 2)

    img2 = np.pad(img, [[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2, pad_w - pad_w // 2], [0, 0]],
                  constant_values=255)
    img3 = img2.reshape(
        img2.shape[0] // tile_size,
        tile_size,
        img2.shape[1] // tile_size,
        tile_size,
        3
    )

    img3 = img3.transpose(0, 2, 1, 3, 4).reshape(-1, tile_size, tile_size, 3)
    n_tiles_with_info = (img3.reshape(img3.shape[0], -1).sum(1) < tile_size ** 2 * 3 * 255).sum()
    if len(img3) < n_tiles:
        img3 = np.pad(img3, [[0, n_tiles - len(img3)], [0, 0], [0, 0], [0, 0]], constant_values=255)
    idxs = np.argsort(img3.reshape(img3.shape[0], -1).sum(-1))[:n_tiles]
    img3 = img3[idxs]
    for i in range(len(img3)):
        result.append({'img': img3[i], 'idx': i})
    return result, n_tiles_with_info >= n_tiles


class PANDADataset(Dataset):
    def __init__(self,
                 df,
                 image_size,
                 n_tiles=n_tiles,
                 tile_mode=0,
                 rand=False,
                 transform=None,
                 ):

        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.n_tiles = n_tiles
        self.tile_mode = tile_mode
        self.rand = rand
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_id = row.image_id

        tiff_file = os.path.join(image_folder, f'{img_id}.tiff')
        image = skimage.io.MultiImage(tiff_file)[1]
        tiles, OK = get_tiles(image, self.tile_mode)

        if self.rand:
            idxes = np.random.choice(list(range(self.n_tiles)), self.n_tiles, replace=False)
        else:
            idxes = list(range(self.n_tiles))

        n_row_tiles = int(np.sqrt(self.n_tiles))
        images = np.zeros((image_size * n_row_tiles, image_size * n_row_tiles, 3))
        for h in range(n_row_tiles):
            for w in range(n_row_tiles):
                i = h * n_row_tiles + w

                if len(tiles) > idxes[i]:
                    this_img = tiles[idxes[i]]['img']
                else:
                    this_img = np.ones((self.image_size, self.image_size, 3)).astype(np.uint8) * 255
                this_img = 255 - this_img
                if self.transform is not None:
                    this_img = self.transform(image=this_img)['image']
                h1 = h * image_size
                w1 = w * image_size
                images[h1:h1 + image_size, w1:w1 + image_size] = this_img

        if self.transform is not None:
            images = self.transform(image=images)['image']
        images = images.astype(np.float32)
        images /= 255
        images = images.transpose(2, 0, 1)

        label = np.zeros(5).astype(np.float32)
        label[:row.isup_grade] = 1.
        return torch.tensor(images), torch.tensor(label)


class enetv2(LightningModule):
    def __init__(self, backbone, out_dim):
        super(enetv2, self).__init__()
        self.enet = enet.EfficientNet.from_pretrained(backbone)

        self.myfc = nn.Linear(self.enet._fc.in_features, out_dim)
        self.enet._fc = nn.Identity()
        self.criterion = nn.BCEWithLogitsLoss()

        valid_idx = np.where((df_train['fold'] == fold))[0]
        self.df_valid = df_train.loc[valid_idx]

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(x)
        return x

    def training_step(self, batch, batch_idx):
        data, target = batch

        logits = self(data)
        loss = self.criterion(logits, target)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        data, target = batch

        logits = self(data)
        loss = self.criterion(logits, target)

        pred = logits.sigmoid().sum(1).detach().round()

        return {"val_loss": loss, "val_true": target.sum(1), "val_pred": pred}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x["val_loss"] for x in outputs])
        targets = torch.cat([x["val_true"] for x in outputs]).detach().cpu().numpy()
        preds = torch.cat([x["val_pred"] for x in outputs]).detach().cpu().numpy()

        val_loss = torch.mean(val_loss).detach().cpu().numpy()

        acc = (preds == targets).mean() * 100.

        qwk = cohen_kappa_score(preds, targets, weights='quadratic')
        df = self.df_valid.iloc[:len(preds)]
        qwk_k = cohen_kappa_score(preds[df['data_provider'] == 'karolinska'],
                                  df[df['data_provider'] == 'karolinska'].isup_grade.values,
                                  weights='quadratic')
        qwk_r = cohen_kappa_score(preds[df['data_provider'] == 'radboud'],
                                  df[df['data_provider'] == 'radboud'].isup_grade.values,
                                  weights='quadratic')
        tensorboard_logs = {"acc": torch.tensor(acc), "val_loss": torch.tensor(val_loss), "qwk": torch.tensor(qwk), "qwk_k": torch.tensor(qwk_k), "qwk_r": torch.tensor(qwk_r)}
        return {"acc": torch.tensor(acc), "val_loss": torch.tensor(val_loss), "qwk": torch.tensor(qwk), "qwk_k": torch.tensor(qwk_k), "qwk_r": qwk_r, "log": torch.tensor(tensorboard_logs)}

    def configure_optimizers(self):
        optimizer = optim.Adam(model.parameters(), lr=init_lr / warmup_factor)

        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs - warmup_epo)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=warmup_factor, total_epoch=warmup_epo,
                                           after_scheduler=scheduler_cosine)

        return [optimizer], [scheduler]

    def train_dataloader(self):
        train_idx = np.where((df_train['fold'] != fold))[0]
        df_this = df_train.loc[train_idx]
        dataset_train = PANDADataset(df_this, image_size, n_tiles, transform=transforms_train)
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                                   sampler=RandomSampler(dataset_train), num_workers=num_workers)
        return train_loader

    def val_dataloader(self):
        dataset_valid = PANDADataset(self.df_valid, image_size, n_tiles, transform=transforms_val)
        valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size,
                                                   sampler=SequentialSampler(dataset_valid), num_workers=num_workers)
        return valid_loader


model = enetv2(enet_type, out_dim=out_dim)

# most basic trainer, uses good defaults
trainer = Trainer(
    gpus=1,
    max_epochs=n_epochs,
    terminate_on_nan=True,
    precision=16 if FP16 else 32,
    checkpoint_callback=ModelCheckpoint(filepath=f"checkpoints/{SAVE_NAME}" + "{epoch}_{kappa:.2f}",
                                        verbose=True, mode="max", monitor="kappa"),
    accumulate_grad_batches=ACCUM_STEPS
)
trainer.fit(model)

torch.save(model.state_dict(), os.path.join(f'{kernel_type}_final_fold{fold}.pth'))
