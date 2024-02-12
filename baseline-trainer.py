import os
import glob
import torch
import argparse

import pytorch_lightning as pl

from torch.utils.data import DataLoader
from lightly.transforms import MSNTransform
from src.data.datsets import BaselinesDataset
from pytorch_lightning.loggers import  CSVLogger

from src.models.baselinemodels import MSN,DINO, MAE



parser = argparse.ArgumentParser(description='SSL training command line interface')


parser.add_argument('--dim',type=int, default=192)
parser.add_argument('--model',type=str, default='msn' )
parser.add_argument('--data-dir',type=str, default=os.path.join('.','data') )
parser.add_argument('--log-dir',type=str, default=os.path.join('.','logs'))
parser.add_argument('--learning-rate', type=float, default=0.0001)
parser.add_argument('--weight-decay', type=float, default=0.001)
parser.add_argument('--max-epochs', type=int, default=50)

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

dataset = BaselinesDataset(data_dir=args.data_dir,
                           transform=transforms)

dataloader = DataLoader(dataset=dataset,
                              batch_size=64,
                              num_workers=24,
                              pin_memory=True,
                              shuffle=True
                              )

if args.model == 'msn':
    model = MSN(image_size=224,
                patch_size=16,
                num_layers=12,
                num_heads=6,
                hiddin_dim=args.dim,
                mask_ratio=0.15,
                num_prototypes=1024,
                lr=args.learning_rate,
                wd=args.weight_decay,
                max_epochs=args.max_epochs)

elif args.model == 'dino':
    model = DINO(image_size=224,
                 patch_size=16,
                 num_layers=12,
                 num_heads=6,
                 hiddin_dim=192,
                 mask_ratio=0.15,
                 num_prototypes=1024,
                 lr=0.00001,
                 wd=0.0001,
                 max_epochs=args.max_epochs)

elif args.models == 'mae':
    model = MAE(image_size=224,
                patch_size=16,
                num_layers=12,
                num_heads=6,
                hiddin_dim=192,
                mask_ratio=0.6,
                decoder_dim=512,
                lr=0.00001,
                wd=0.0001,
                max_epochs=args.max_epochs)


trainer = pl.Trainer(max_epochs=args.max_epochs,
                     log_every_n_steps=1,
                     precision='16-mixed', 
                     default_root_dir=args.log_dir
                     )

torch.set_float32_matmul_precision('medium')
trainer.fit(model=model, train_dataloaders=dataloader)