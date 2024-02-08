import os
import glob
import torch
import argparse
import pytorch_lightning as pl
import torch.nn as nn
from torchvision.models.vision_transformer import VisionTransformer
from src.models.evaluationmodels import EvaluationModel
from src.data.datsets import MIMICCXR,NIHDataset
from src.data.utils import train_transforms, val_test_transforms
from torch.utils.data import  DataLoader
from src.models.utils import parse_weights
import torchvision
from transformers import ViTForImageClassification
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
pl.seed_everything(24)

data_dir = os.getenv('DATA_DIR')


parser = argparse.ArgumentParser(description='SSL evaluation command line interface')




# data loading options

parser.add_argument('--data-percent','--dp', type=float, default=0.05, metavar='PERCENT',
                    help='dataset percentage | default: (0.05)'
                   )

# model options


parser.add_argument('--learning-rate','--lr', type=float, default=0.0001,
                    help='model learning rate | default: (0.01)'
                   )

parser.add_argument('-f','--freeze', type=int, default=1, 
                    help='enables linear evaluation | default: (True)'
                   )
parser.add_argument('-s','--scheduler', type=str, default='cosine',
                    help='enables linear evaluation | default: (True)'
                   )
parser.add_argument('-d','--dim', type=int, default=192, 
                    help='enables linear evaluation | default: (True)'
                   )

args = parser.parse_args()


# paths = glob.glob(os.path.join(data_dir,'resized','**','*.jpg'), recursive=True)
# train_dataset = MIMICCXR(paths=paths,
#                          data_dir= data_dir, 
#                          split='train', 
#                          transform=train_transforms,
#                          percentage=float(args.data_percent)
#                          )
# val_dataset = MIMICCXR(paths=paths,
#                         data_dir=data_dir, 
#                         split='validate', 
#                         transform=val_test_transforms,
#                         )
# test_dataset = MIMICCXR(paths=paths,
#                         data_dir=data_dir, 
#                         split='test', 
#                         transform=val_test_transforms,
#                         )


train_dataset = NIHDataset(root=data_dir,
                           data_path=data_dir,
                           transform=train_transforms)

val_dataset = NIHDataset(root=data_dir,
                           data_path=data_dir,
                           transform=val_test_transforms)

test_dataset = NIHDataset(root=data_dir,
                           data_path=data_dir,
                           transform=val_test_transforms)


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

# backbone = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')

checkpoint_dir = os.getenv('CKPT_PATH')
all_weights = torch.load(checkpoint_dir,map_location='cpu')['state_dict']

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

logger = CSVLogger(save_dir= f'/scratch/sas10092/ChexMSN/notebooks/nih/{os.getenv("VARS")}/{os.getenv("BACKBONE")}',
                    version=os.getenv('SLURM_JOB_ID'))

model = EvaluationModel(backbone=backbone,
                        learning_rate=args.learning_rate,
                        weight_decay=0.0,
                        output_dim=14,
                        freeze=args.freeze,
                        max_epochs=50,
                        scheduler=args.scheduler,
                        summary_path=f'/scratch/sas10092/ChexMSN/notebooks/nih/{os.getenv("VARS")}/{os.getenv("BACKBONE")}/lightning_logs/{logger.version}')

model.backbone.heads.head= nn.Linear(in_features=model.backbone.heads.head.in_features,
                                      out_features=model.output_dim)


trainer = pl.Trainer(max_epochs=50,
                     num_sanity_val_steps=0,
                     callbacks=[checkpoint_callback,early_stop],
                     logger=logger,
                     )


print(args.freeze,args.learning_rate,args.dim,args.scheduler)
trainer.fit(model=model, train_dataloaders=train_dataloader,val_dataloaders=val_dataloader)
trainer.test(model=model,dataloaders=test_dataloader,ckpt_path='best')