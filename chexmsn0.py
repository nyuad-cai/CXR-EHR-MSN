import os
import glob
import torch
import pytorch_lightning as pl
from src.data.datsets import PretrainDataset, MIMICVal
from torch.utils.data import DataLoader

from src.data.utils import IMAGENET_STAT, val_test_transforms
from src.models.chexmsn import MSN, MSN1
from pytorch_lightning.loggers import  CSVLogger



data_dir = os.getenv('DATA_DIR')
train_dataset = PretrainDataset(data_dir=os.path.join(data_dir,'resized'))
train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=64,
                              num_workers=16,
                              pin_memory=True,
                              shuffle=True
                              )


paths = paths = glob.glob(os.path.join(data_dir,'resized','**','*.jpg'), recursive=True)
val_dataset = MIMICVal(paths=paths,
                       data_dir=data_dir,
                       split='validate',
                       transform = val_test_transforms)

val_dataloader = DataLoader(dataset= val_dataset,
                            batch_size=64,
                            drop_last=True,
                            num_workers=16,
                            pin_memory=True
                            )

# model = MSN(dataloader_kNN=val_dataloader,
#            num_classes=2,
#            knn_k=20,
#            knn_t=0.2,
#            mask_ratio=0.15,
#            lr=0.1,
#            prototypes_num=1024,
#            weight_decay=0.0,
#            max_epochs=100
#            )

model = MSN1()

trainer = pl.Trainer(max_epochs=100,
                     log_every_n_steps=1,
                     precision=16, 
                     default_root_dir='./models/'
                     
                     )

torch.set_float32_matmul_precision('medium')

trainer.fit(model=model, train_dataloaders=train_dataloader)