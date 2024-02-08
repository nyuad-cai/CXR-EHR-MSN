import os
import glob
import torch
import pytorch_lightning as pl
from src.data.datsets import BaselinesDataset
from torch.utils.data import DataLoader

from src.data.utils import IMAGENET_STAT, val_test_transforms
from src.models.baselinemodels import DINO, MAE
from pytorch_lightning.loggers import  CSVLogger

data_dir = os.getenv('DATA_DIR')

dataset = BaselinesDataset(data_dir=data_dir)
dataloader = DataLoader(dataset=dataset,
                              batch_size=64,
                              num_workers=24,
                              pin_memory=True,
                              shuffle=True
                              )

model = MAE()


trainer = pl.Trainer(max_epochs=100,
                     log_every_n_steps=1,
                     precision='16-mixed', 
                     default_root_dir='./notebooks/'
                     )

torch.set_float32_matmul_precision('medium')
trainer.fit(model=model, train_dataloaders=dataloader)