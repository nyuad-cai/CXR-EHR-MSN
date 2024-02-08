import os
import copy
import torch
import torchvision

import torch.nn as nn
from typing import List
import pytorch_lightning as pl
from lightly.loss import MSNLoss
from src.models.utils import mask_tensor
from lightly.models.modules.heads import MSNProjectionHead
from lightly.models.modules.masked_autoencoder import MAEBackbone



import lightly.models.utils as utils



class MSN1(pl.LightningModule):
    def __init__(self,
                 lr=0.0001,
                 wd=0.001):
        super().__init__()

        self.mask_ratio = 0.15
        self.lr = lr
        self.wd = wd
        hd = 192
        ehr = int(os.getenv('EHR'))
        ehr_emb = 128
        # ehr embedding layer
        self.ehr_embed = nn.Linear(in_features=ehr,out_features=ehr_emb)
        
        # cxr-ehr project layer
        self.ehr_cxr_project = nn.Linear(in_features=ehr_emb+hd,out_features=hd)
        
        vit = torchvision.models.VisionTransformer(image_size=224,
                                                   patch_size=16,
                                                   num_layers=12,
                                                   num_heads=6,
                                                   hidden_dim=hd,
                                                   mlp_dim=hd*4)
        
        self.backbone = MAEBackbone.from_vit(vit)
        self.projection_head = MSNProjectionHead(hd,hd*4,hd)

        self.anchor_backbone = copy.deepcopy(self.backbone)
        self.anchor_projection_head = copy.deepcopy(self.projection_head)

        utils.deactivate_requires_grad(self.backbone)
        utils.deactivate_requires_grad(self.projection_head)

        self.prototypes = nn.Linear(hd, 1024, bias=False).weight
        self.criterion = MSNLoss()

    def training_step(self, batch, batch_idx):
        utils.update_momentum(self.anchor_backbone, self.backbone, 0.996)
        utils.update_momentum(self.anchor_projection_head, self.projection_head, 0.996)

        views, ehr = batch[0], batch[1]
        views = [view.to(self.device, non_blocking=True) for view in views]
        ehr = ehr.to(self.device)
        
        targets = views[0]
        anchors = views[1]
        anchors_focal = torch.concat(views[2:], dim=0)
        
        ehr_embed = self.ehr_embed(ehr)
        ehr_embed = ehr_embed.repeat((11,1))


        targets_out = self.backbone(targets)
        targets_out = self.projection_head(targets_out)
        
        anchors_out = self.encode_masked(anchors)
        anchors_focal_out = self.encode_masked(anchors_focal)
        anchors_out = torch.cat([anchors_out, anchors_focal_out], dim=0)
        anchors_out = self.ehr_cxr_project(torch.cat((anchors_out,ehr_embed),1))
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
                                                               T_max=100)
        return {'optimizer': optimizer,
               'lr_scheduler': scheduler
               }




class MSN2(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.mask_ratio = 0.15
        hd = 384
        # ehr embedding layer
        self.ehr_embed = nn.Linear(in_features=3,out_features=64)
        
        # cxr-ehr project layer
        self.ehr_cxr_project = nn.Linear(in_features=64+hd,out_features=hd)
        
        vit = torchvision.models.VisionTransformer(image_size=224,
                                                   patch_size=16,
                                                   num_layers=12,
                                                   num_heads=6,
                                                   hidden_dim=hd,
                                                   mlp_dim=hd*4)
        
        self.backbone = MAEBackbone.from_vit(vit)
        self.projection_head = MSNProjectionHead(hd,hd*4,hd)

        self.anchor_backbone = copy.deepcopy(self.backbone)
        self.anchor_projection_head = copy.deepcopy(self.projection_head)
        self.anchor_ehr_cxr_project = copy.deepcopy(self.ehr_cxr_project)

        utils.deactivate_requires_grad(self.backbone)
        utils.deactivate_requires_grad(self.projection_head)
        utils.deactivate_requires_grad(self.ehr_cxr_project)
        
        self.prototypes = nn.Linear(hd, 1024, bias=False).weight
        self.criterion = MSNLoss()

    def training_step(self, batch, batch_idx):
        utils.update_momentum(self.anchor_backbone, self.backbone, 0.996)
        utils.update_momentum(self.anchor_projection_head, self.projection_head, 0.996)
        utils.update_momentum(self.anchor_ehr_cxr_project, self.ehr_cxr_project, 0.996)
        
        views, ehr = batch[0], batch[1]
        ehr_masked = mask_tensor(ehr)
        views = [view.to(self.device, non_blocking=True) for view in views]
        ehr = ehr.to(self.device)
        ehr_masked = ehr_masked.to(self.device)
        
        targets = views[0]
        anchors = views[1]
        anchors_focal = torch.concat(views[2:], dim=0)
        
        ehr_embed = self.ehr_embed(ehr)
        ehr_masked_embed = self.ehr_embed(ehr_masked)
        ehr_masked_embed = ehr_embed.repeat((11,1))
        

        targets_out = self.backbone(targets)
        targets_out = self.ehr_cxr_project(torch.cat((targets_out,ehr_embed),1))
        targets_out = self.projection_head(targets_out)
        
        anchors_out = self.encode_masked(anchors)
        anchors_focal_out = self.encode_masked(anchors_focal)
        anchors_out = torch.cat([anchors_out, anchors_focal_out], dim=0)
        anchors_out = self.anchor_ehr_cxr_project(torch.cat((anchors_out,ehr_masked_embed),1))
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
                                      lr=0.0001,
                                      weight_decay=0.001)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                               eta_min=0.0,
                                                               T_max=100)
        return {'optimizer': optimizer,
               'lr_scheduler': scheduler
               }



    



    


