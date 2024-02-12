import os
import torch
import argparse

import pytorch_lightning as pl

from torch.utils.data import DataLoader
from src.data.datsets import CxrEhrDataset
from src.models.cxrehrmsn import CxrEhrMSN
from pytorch_lightning.loggers import  CSVLogger

from lightly.transforms import MSNTransform
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

pl.seed_everything(24)
torch.set_float32_matmul_precision('high')


parser = argparse.ArgumentParser(description='SSL training command line interface')


parser.add_argument('--dim',type=int, default=192)
parser.add_argument('--ehr-in',type=int, default=2)
parser.add_argument('--ehr-out',type=int, default=128)
parser.add_argument('--data-dir',type=str, default=os.path.join('./','data','meta.csv'))
parser.add_argument('--log-dir',type=str, default=os.path.join('.','logs'))


parser.add_argument('--num-prototypes', type=int, default=1024)
parser.add_argument('--learning-rate', type=float, default=0.0001)
parser.add_argument('--weight-decay', type=float, default=0.001)
parser.add_argument('--max-epochs', type=int, default=100)

args = parser.parse_args()



transforms = MSNTransform(random_size=224,
                          focal_size=96,
                          random_views=2,
                          focal_views=10,
                          random_crop_scale=(0.3,1),
                          cj_prob=0.0,
                          gaussian_blur=0.0,
                          random_gray_scale=0.0,
                          hf_prob=0.5)

dataset =CxrEhrDataset(data_dir=os.getenv('DATA_DIR'),
                         transforms= transforms)

dataloader = DataLoader(dataset=dataset,
                              batch_size=64,
                              num_workers=24,
                              pin_memory=True,
                              shuffle=True
                              )


model = CxrEhrMSN(image_size=224,
                  patch_size=16,
                  num_layers=12,
                  num_heads=6,
                  hiddin_dim=args.dim,
                  ehr_in=args.ehr_in,
                  ehr_out=args.ehr_out,
                  num_prototypes=args.num_,
                  lr=args.learning_rate,
                  wd=args.wd,
                  mask_ratio=0.15,
                  max_epochs= args.max_epochs
                  )

checkpoint_callback = ModelCheckpoint(monitor='train_loss', 
                                      mode='min',
                                      every_n_epochs=1,
                                      save_top_k=1,
                                      )

early_stop = EarlyStopping(monitor='train_loss', 
                           min_delta=0.00001,
                           mode='min', 
                           patience=5)

csv_logger = CSVLogger(save_dir=args.log_dir,
                       flush_logs_every_n_steps=1)


trainer = pl.Trainer(accelerator='auto', 
                     devices='auto',
                     strategy='auto',
                     logger=csv_logger, 
                     log_every_n_steps=1,
                     max_epochs=args.max_epochs,
                     precision='16-mixed', 
                     callbacks=[checkpoint_callback,early_stop],
                     default_root_dir=args.log_dir
                     )



trainer.fit(model=model, train_dataloaders=dataloader)