import os
import random
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.nn import Linear, Sequential, Dropout, ReLU, Conv2d, BatchNorm2d
from torch.nn import functional as F
from torchvision import models
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint

import sys
sys.path.append("../")
from utils.dataset import TrainDataset

import albumentations as A

import warnings
warnings.filterwarnings("ignore")

NUM_EPOCHS = 32
SEED = 0
BATCH_SIZE = 3
BASE_BATCH = 32
LR = 1e-4
MIN_LR = 1e-6
WEIGHT_DECAY = 5e-5
FP16 = False
SAVE_NAME = "checkpoints/resnet50_128_100"
ACCUM_STEPS = 1

mean = [127.66098, 127.66102, 127.66085]
std = [10.5911, 10.5911045, 10.591107]


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(SEED)


class Model(LightningModule):
    def __init__(self, outputs=6):
        super().__init__()

        self.net = models.resnet50(True)
        self.linear = Sequential(ReLU(), Dropout(),  Linear(1000, outputs))

        df = pd.read_csv("/home/dipet/kaggle/prostate/input/prostate-cancer-grade-assessment/train.csv")
        self.train_df, self.valid_df = train_test_split(df, test_size=0.2)
        self.data_dir = "/datasets/panda/train_128_100"

        self.train_transforms = A.Compose(
            [
                A.InvertImg(p=1),
                A.RandomGridShuffle(grid=(10, 10)),
                A.RandomScale(0.1),
                A.PadIfNeeded(1280, 1280),
                A.RandomSizedCrop([1000, 1280], 1280, 1280),
                A.Flip(),
                A.Rotate(90),
                A.RandomBrightnessContrast(0.02, 0.02),
                A.HueSaturationValue(0, 10, 10),
                A.Normalize(mean, std, 1),
            ]
        )
        self.valid_transforms = A.Compose([A.InvertImg(p=1), A.Normalize(mean, std, 1),])

    def forward(self, x):
        x = self.net(x)
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {
            "train_loss": loss,
        }
        if torch.isnan(loss) or not torch.isfinite(loss):
            self(x)
            print()
        return {"loss": loss, "log": tensorboard_logs}

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, eps=5e-5)
        scheduler = ReduceLROnPlateau(opt, factor=0.5, patience=3, min_lr=MIN_LR, verbose=True)
        return [opt], [scheduler]

    def train_dataloader(self):
        dataset = TrainDataset(self.train_df, self.data_dir, self.train_transforms)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=12, shuffle=True)
        return loader

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return {"val_loss": F.cross_entropy(y_hat, y), "y_true": y, "y_pred": y_hat}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        y = torch.cat([x["y_true"] for x in outputs])
        y_hat = torch.cat([x["y_pred"] for x in outputs], 0)
        y_hat = torch.argmax(y_hat, 1)
        kappa = cohen_kappa_score(y.detach().cpu().numpy(), y_hat.detach().cpu().numpy(), weights="quadratic")
        tensorboard_logs = {"val_loss": avg_loss, "kappa": kappa}
        print(f"val_loss: {avg_loss}, kappa: {kappa}")
        return {"val_loss": avg_loss, "log": tensorboard_logs, "kappa": torch.tensor(kappa)}

    def val_dataloader(self):
        dataset = TrainDataset(self.valid_df, self.data_dir, self.valid_transforms)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=12)
        return loader


model = Model()

# most basic trainer, uses good defaults
trainer = Trainer(
    gpus=1,
    max_epochs=NUM_EPOCHS,
    terminate_on_nan=True,
    precision=16 if FP16 else 32,
    checkpoint_callback=ModelCheckpoint(filepath=f"{SAVE_NAME}" + "_{epoch}_{kappa:.2f}", verbose=True, mode="max", monitor="kappa"),
    accumulate_grad_batches=ACCUM_STEPS,
    gradient_clip_val=0.5
)
trainer.fit(model)
