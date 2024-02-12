import os
import glob
import torch
import argparse
import pytorch_lightning as pl
import torch.nn as nn

from torchvision.models.vision_transformer import VisionTransformer
from src.models.evaluationmodels import EvaluationModel
from src.data.datsets import MIMICCXR,CheXpertDataset, NIHDataset
from src.data.utils import train_transforms, val_test_transforms
from torch.utils.data import  DataLoader
from src.models.utils import parse_weights
import torchvision
from transformers import ViTForImageClassification
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
pl.seed_everything(24)




parser = argparse.ArgumentParser(description='SSL evaluation command line interface')

parser.add_argument('--dim', type=int, default=192)
parser.add_argument('--freeze', type=int, default=1)
parser.add_argument('--dataset', type=str, default='mimic')
parser.add_argument('--log-dir', type=str, default='./logs')
parser.add_argument('--scheduler', type=str, default='cosine')
parser.add_argument('--data-percent', type=float, default=1.0)
parser.add_argument('--learning-rate', type=float, default=0.0001)
parser.add_argument('--max-epochs', type=int, default=100)

args = parser.parse_args()

if args.dataset == 'mimic':
    mimic_data_dir = ''
    paths = glob.glob(os.path.join(mimic_data_dir,'resized','**','*.jpg'), recursive=True)
    train_dataset = MIMICCXR(paths=paths,
                         data_dir= mimic_data_dir, 
                         split='train', 
                         transform=train_transforms,
                         percentage=float(args.data_percent))
    
    val_dataset = MIMICCXR(paths=paths,
                           data_dir=mimic_data_dir, 
                           split='validate', 
                           transform=val_test_transforms)
    
    test_dataset = MIMICCXR(paths=paths,
                            data_dir=mimic_data_dir, 
                            split='test', 
                            transform=val_test_transforms)



elif args.dataset == 'nih':
    nih_data_dir = ''
    nih_train_labels = ''
    nih_val_labels = ''
    nih_test_labels = ''
    train_dataset = NIHDataset(root=nih_data_dir,
                             data_path=nih_train_labels,
                             transform=train_transforms)

    val_dataset = NIHDataset(root=nih_data_dir,
                             data_path=nih_val_labels,
                             transform=val_test_transforms)

    test_dataset = NIHDataset(root=nih_data_dir,
                            data_path=nih_test_labels,
                            transform=val_test_transforms)
    

elif args.dataset == 'chexpert':
    chexpert_data_dir = ''
    ckpt_path = ''
    chexpert_dataset = CheXpertDataset(data_path=chexpert_data_dir,
                               transform=val_test_transforms)

    chexpert_dataloader = DataLoader(dataset=chexpert_dataset,
                                     batch_size=len(chexpert_dataset),
                                     shuffle=False,
                                     num_workers=24,
                                     pin_memory=True)
    

train_dataloader = DataLoader(dataset=train_dataset,
                             batch_size=64,
                             shuffle=True,
                             num_workers=24,
                             pin_memory=True,
                             drop_last=True
                             )

val_dataloader = DataLoader(dataset=val_dataset,
                             batch_size=64,
                             shuffle=False,
                             num_workers=24,
                             pin_memory=True,
                             drop_last=True
                             )

test_dataloader = DataLoader(dataset=test_dataset,
                             batch_size=64,
                             shuffle=False,
                             num_workers=24,
                             pin_memory=True,
                             drop_last=True
                             )



backbone = VisionTransformer(image_size=224,
                             patch_size=16,
                             num_layers=12,
                             num_heads=6,
                             hidden_dim=args.dim,
                             mlp_dim=args.dim*4)



ckpt_path = ''
all_weights = torch.load(ckpt_path,map_location='cpu')['state_dict']
weight = parse_weights(all_weights)

msg = backbone.load_state_dict(weight,strict=False)
print(msg)

early_stop = EarlyStopping(monitor='val_auroc', 
                           min_delta=0.00001,
                           mode='max', 
                           patience=5
                          )

checkpoint_callback = ModelCheckpoint(monitor='val_auroc', 
                                      mode='max',
                                      every_n_epochs=1,
                                      save_top_k=1,
                                     )

logger = CSVLogger(save_dir= args.log_dir)


model = EvaluationModel(backbone=backbone,
                        learning_rate=args.learning_rate,
                        output_dim=14,
                        freeze=args.freeze,
                        max_epochs=args.max_epochs,
                        scheduler=args.scheduler,
                        summary_path=os.path.join(args.log_dir,'lightning_logs',logger.version))

model.backbone.heads.head= nn.Linear(in_features=model.backbone.heads.head.in_features,
                                      out_features=model.output_dim)


trainer = pl.Trainer(max_epochs=args.max_epochs,
                     num_sanity_val_steps=0,
                     callbacks=[checkpoint_callback,early_stop],
                     logger=logger,
                     )

if args.dataset in ['mimic','nih']:

    trainer.fit(model=model, train_dataloaders=train_dataloader,val_dataloaders=val_dataloader)
    trainer.test(model=model,dataloaders=test_dataloader,ckpt_path='best')

elif args.dataset == 'chexpert':
    trainer.test(model=model,dataloaders=chexpert_dataloader,ckpt_path=ckpt_path)