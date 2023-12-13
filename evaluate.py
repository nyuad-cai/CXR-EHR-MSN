import os
import glob
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torchvision.models.vision_transformer import VisionTransformer
from src.models.evaluationmodels import EvaluationModel
from src.data.datsets import MIMICCXR
from src.data.utils import train_transforms, val_test_transforms
from torch.utils.data import  DataLoader
from src.models.utils import  parse_weights
from torchvision.models import vit_b_16
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

data_dir = os.getenv('DATA_DIR')

paths = glob.glob(os.path.join(data_dir,'resized','**','*.jpg'), recursive=True)
train_dataset = MIMICCXR(paths=paths,
                         data_dir= data_dir, 
                         split='train', 
                         transform=train_transforms,
                         )
val_dataset = MIMICCXR(paths=paths,
                        data_dir=data_dir, 
                        split='validate', 
                        transform=val_test_transforms,
                        )
test_dataset = MIMICCXR(paths=paths,
                        data_dir=data_dir, 
                        split='test', 
                        transform=val_test_transforms,
                        )


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
                             hidden_dim=768,
                             mlp_dim=768*4)



# checkpoint_dir = os.getenv('CKPT_PATH')
# all_weights = torch.load(checkpoint_dir,map_location='cpu')['state_dict']

# weight = parse_weights(all_weights)

# msg = backbone.load_state_dict(weight,strict=False)
# print(msg)

checkpoint_callback = ModelCheckpoint(monitor='val_auroc', 
                                      mode='max',
                                      every_n_epochs=1,
                                      save_top_k=1,
                                     )
early_stop = EarlyStopping(monitor='val_auroc', 
                           min_delta=0.0001,
                           mode='max', 
                           patience=4
                          )

model = EvaluationModel(backbone=backbone,
                        learning_rate=0.0005,
                        weight_decay=0.001,
                        output_dim=14,
                        freeze=False,
                        max_epochs=50)

model.backbone.heads.head = nn.Linear(in_features=model.backbone.heads.head.in_features,
                                      out_features=model.output_dim)


trainer = pl.Trainer(max_epochs=50,
                     num_sanity_val_steps=0,
                     default_root_dir='/scratch/sas10092/ChexMSN/notebooks/',
                     callbacks=[checkpoint_callback, early_stop],
                     )



trainer.fit(model=model, train_dataloaders=train_dataloader,val_dataloaders=val_dataloader)
trainer.test(model=model,dataloaders=test_dataloader)