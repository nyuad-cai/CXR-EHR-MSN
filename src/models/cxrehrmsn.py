import os
import copy
import torch
import torchvision

import torch.nn as nn
import pytorch_lightning as pl
import lightly.models.utils as utils

from lightly.loss import MSNLoss
from lightly.models.modules.heads import MSNProjectionHead
from lightly.models.modules.masked_autoencoder import MAEBackbone







class CxrEhrMSN(pl.LightningModule):
    def __init__(self,
                 image_size: int = 224,
                 patch_size: int = 16,
                 num_layers: int = 12,
                 num_heads: int = 6,
                 hiddin_dim: int = 192,
                 ehr_in: int = 2,
                 ehr_out: int = 128,
                 mask_ratio: float = 0.15,
                 num_prototypes: int = 1024,
                 lr: float = 0.0001,
                 wd=0.001,
                 max_epochs: int = 100):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hiddin_dim

        vit = torchvision.models.VisionTransformer(image_size=self.image_size,
                                                   patch_size=self.patch_size,
                                                   num_layers=self.num_layers,
                                                   num_heads=self.num_heads,
                                                   hidden_dim=self.hidden_dim,
                                                   mlp_dim=self.hidden_dim*4)
        
        self.backbone = MAEBackbone.from_vit(vit)
        self.projection_head = MSNProjectionHead(self.hidden_dim,self.hidden_dim*4,self.hidden_dim)

        self.anchor_backbone = copy.deepcopy(self.backbone)
        self.anchor_projection_head = copy.deepcopy(self.projection_head)
        utils.deactivate_requires_grad(self.backbone)
        utils.deactivate_requires_grad(self.projection_head)

        self.num_prototypes = num_prototypes
        self.prototypes = nn.Linear(self.hidden_dim, self.num_prototypes, bias=False).weight
        self.criterion = MSNLoss()        
        
        

        self.ehr_in = ehr_in
        self.ehr_out = ehr_out
        # ehr embedding layer
        self.f_ehr = nn.Linear(in_features=self.ehr_in,out_features=self.ehr_out)
        
        # cxr-ehr project layer
        self.ehr_cxr_project_g = nn.Linear(in_features=self.ehr_out+self.hidden_dim,out_features=self.hidden_dim)
        
        self.mask_ratio = mask_ratio
        self.lr = lr
        self.wd = wd
        self.max_epochs = max_epochs
        
    def training_step(self, batch, batch_idx):
        utils.update_momentum(self.anchor_backbone, self.backbone, 0.996)
        utils.update_momentum(self.anchor_projection_head, self.projection_head, 0.996)

        views, ehr = batch[0], batch[1]
        views = [view.to(self.device, non_blocking=True) for view in views]
        ehr = ehr.to(self.device)
        
        targets = views[0]
        anchors = views[1]
        anchors_focal = torch.concat(views[2:], dim=0)
        
        ehr_embed = self.f_ehr(ehr)
        ehr_embed = ehr_embed.repeat((11,1))


        targets_out = self.backbone(targets)
        targets_out = self.projection_head(targets_out)
        
        anchors_out = self.encode_masked(anchors)
        anchors_focal_out = self.encode_masked(anchors_focal)
        anchors_out = torch.cat([anchors_out, anchors_focal_out], dim=0)
        anchors_out = self.ehr_cxr_project_g(torch.cat((anchors_out,ehr_embed),1))
        anchors_out = self.anchor_projection_head(anchors_out)
        
        loss = self.criterion(anchors_out, targets_out, self.prototypes.data) 
        self.log("train_loss", loss, on_epoch= True,on_step=True , logger=True, prog_bar=True)
        
        return loss


    def encode_masked(self, anchors):
        batch_size, _, _, width = anchors.shape
        seq_length = (width // self.anchor_backbone.patch_size) ** 2
        idx_keep, _ = utils.random_token_mask(
            size=(batch_size, seq_length),
            mask_ratio=self.mask_ratio,
            device=self.device,
        )
        out = self.anchor_backbone(anchors, idx_keep=idx_keep)
        return out

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.parameters(), 
                                      lr=self.lr,
                                      weight_decay=self.wd)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                               eta_min=0.0,
                                                               T_max=self.max_epochs)
        return {'optimizer': optimizer,
               'lr_scheduler': scheduler
               }








    



    


