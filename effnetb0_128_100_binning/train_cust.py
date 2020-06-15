import os
import random
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.nn import Linear, Sequential, Dropout, ReLU, BCEWithLogitsLoss, Module
from torchvision import models
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint

from efficientnet_pytorch import EfficientNet

from tqdm import tqdm

import sys
sys.path.append("../")
from utils.dataset import TrainDatasetBinning

import albumentations as A

import warnings
warnings.filterwarnings("ignore")

NUM_EPOCHS = 32
SEED = 0
BATCH_SIZE = 2
ACCUM_STEPS = 1
LR = 3e-4
MIN_LR = 2e-5
WEIGHT_DECAY = 2e-5
FP16 = True

mean = [127.66098, 127.66102, 127.66085]
std = [10.5911, 10.5911045, 10.591107]

NUM_WORKERS = 12

SAVE_NAME = "effnetb0_128_100_binning"


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(SEED)


class Model(Module):
    def __init__(self, outputs=5):
        super().__init__()
        self.net = EfficientNet.from_pretrained('efficientnet-b0')
        self.linear = Sequential(ReLU(), Dropout(),  Linear(1000, outputs))

        df = pd.read_csv("../input/prostate-cancer-grade-assessment/train.csv")
        self.train_df, self.valid_df = train_test_split(df, test_size=0.2)
        self.data_dir = "/datasets/panda/train_128_100"

        self.train_transforms = A.Compose(
            [
                A.InvertImg(p=1),
                A.RandomSizedCrop([100, 128], 128, 128),
                A.Transpose(),
                A.Flip(),
                A.Rotate(90),
                A.Normalize(mean, std, 1),
            ]
        )
        self.valid_transforms = A.Compose([A.InvertImg(p=1), A.Normalize(mean, std, 1),])

        self.criterion = BCEWithLogitsLoss()

    def forward(self, x):
        x = self.net(x)
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        if torch.isnan(loss):
            print()
        return {"loss": loss, "train_y_true": y, "train_y_pred": y_hat}

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, eps=2e-5)
        # scheduler = ReduceLROnPlateau(opt, factor=0.5, patience=3, min_lr=MIN_LR, verbose=True)
        scheduler = CosineAnnealingLR(opt, NUM_EPOCHS, MIN_LR)
        return [opt], [scheduler]

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        y = torch.cat([x["train_y_true"] for x in outputs])
        y = y.sum(1)

        y_hat = torch.cat([x["train_y_pred"] for x in outputs], 0)
        y_hat = torch.round(y_hat.sigmoid().sum(1))
        kappa = cohen_kappa_score(y.detach().cpu().numpy(), y_hat.detach().cpu().numpy(), weights="quadratic")
        tensorboard_logs = {"loss": avg_loss, "train_kappa": kappa}
        return {"loss": avg_loss, "log": tensorboard_logs, "train_kappa": torch.tensor(kappa)}

    def train_dataloader(self):
        dataset = TrainDatasetBinning(self.train_df, self.data_dir, self.train_transforms, 1, 128, 100)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
        return loader

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return {"val_loss": self.criterion(y_hat, y), "y_true": y, "y_pred": y_hat}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        y = torch.cat([x["y_true"] for x in outputs])
        y = y.sum(1)

        y_hat = torch.cat([x["y_pred"] for x in outputs], 0)
        y_hat = torch.round(y_hat.sigmoid().sum(1))
        kappa = cohen_kappa_score(y.detach().cpu().numpy(), y_hat.detach().cpu().numpy(), weights="quadratic")
        tensorboard_logs = {"val_loss": avg_loss, "kappa": kappa}
        print(f"val_loss: {avg_loss}, kappa: {kappa}")
        return {"val_loss": avg_loss, "log": tensorboard_logs, "kappa": torch.tensor(kappa)}

    def val_dataloader(self):
        dataset = TrainDatasetBinning(self.valid_df, self.data_dir, self.valid_transforms, 1, 128, 100, random=False)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        return loader


def train_epoch(model: Model, loader, optimizer):
    model.train()
    outputs = []
    bar = tqdm(loader)
    for i, (data, target) in enumerate(bar):
        data, target = data.cuda(), target.cuda()
        if ACCUM_STEPS % i == 0:
            optimizer.zero_grad()
        outputs.append(model.training_step((data, target), i))
        loss = outputs[-1]['loss']
        loss.backward()
        if ACCUM_STEPS % i == (ACCUM_STEPS - 1):
            optimizer.step()

        loss_np = loss.detach().cpu().numpy()
        bar.set_description('loss: %.5f' % loss_np)
    out = model.training_epoch_end(outputs)

    return out['loss'], out['train_kappa']


def val_epoch(model, loader):
    model.eval()
    outputs = []

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(loader)):
            data, target = data.cuda(), target.cuda()
            outputs.append(model.validation_step((data, target), i))

    out = model.validation_epoch_end(outputs)

    return out['val_loss'], out['kappa']


model = Model()

train_loader = model.train_dataloader()
valid_loader = model.val_dataloader()

opt, sched = model.configure_optimizers()
opt = opt[0]
sched = sched[0]
best_kappa = -float('inf')

import time

model.cuda()
for epoch in range(1, NUM_EPOCHS):
    print(time.ctime(), 'Epoch:', epoch)
    sched.step(epoch - 1)

    train_loss, tr_kappa = train_epoch(model, train_loader, opt)
    val_loss, kappa = val_epoch(model, valid_loader)

    content = time.ctime() + f' Epoch {epoch}, lr: {opt.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.5f}, val loss: {val_loss:.5f}, train_kappa: {tr_kappa:.5f}, kappa: {kappa:.5f}'
    print(content)
    with open(f'log.txt', 'a') as appender:
        appender.write(content + '\n')

    if kappa > best_kappa:
        prev = best_kappa
        best_kappa = kappa
        save_path = f"checkpoints/{SAVE_NAME}_{epoch}_{best_kappa:.2f}.pth"
        print(f"Kappa: {best_kappa} best then {prev}. Save model to {save_path}")
        torch.save(model.state_dict(), save_path)

    torch.save(model.state_dict(), f"checkpoints/{SAVE_NAME}_last.pth")
