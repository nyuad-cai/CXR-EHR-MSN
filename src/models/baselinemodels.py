import copy
import torch

import torch.nn as nn
import pytorch_lightning as pl
import lightly.models.utils as utils

from lightly.loss import MSNLoss, DINOLoss
from lightly.utils.scheduler import cosine_schedule
from lightly.models.modules.masked_autoencoder import MAEBackbone, MAEDecoder
from lightly.models.modules.heads import MSNProjectionHead, DINOProjectionHead


# vanilla MSN
class MSN(pl.LightningModule):
    def __init__(self,
                 image_size: int = 224,
                 patch_size: int = 16,
                 num_layers: int = 12,
                 num_heads: int = 6,
                 hiddin_dim: int = 192,
                 mask_ratio: float = 0.15,
                 num_prototypes: int = 1024,
                 lr: float = 0.00001,
                 wd=0.0001,
                 max_epochs: int = 100):
        
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hiddin_dim
        self.mask_ratio = mask_ratio
        self.num_prototypes = num_prototypes
        self.lr = lr
        self.wd = wd
        self.max_epochs = max_epochs

        # ViT small configuration (ViT-S/16)
        self.backbone = MAEBackbone(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            hidden_dim=self.hidden_dim,
            mlp_dim=self.hidden_dim * 4)
        
        
        self.projection_head = MSNProjectionHead(self.hidden_dim,self.hidden_dim*4,self.hidden_dim)

        self.anchor_backbone = copy.deepcopy(self.backbone)
        self.anchor_projection_head = copy.deepcopy(self.projection_head)

        utils.deactivate_requires_grad(self.backbone)
        utils.deactivate_requires_grad(self.projection_head)

        self.prototypes = nn.Linear(self.hidden_dim, self.num_prototypes, bias=False).weight
        self.criterion = MSNLoss()


    def training_step(self, batch, batch_idx):
        utils.update_momentum(self.anchor_backbone, self.backbone, 0.996)
        utils.update_momentum(self.anchor_projection_head, self.projection_head, 0.996)

        views = batch[0]
        views = [view.to(self.device, non_blocking=True) for view in views]
        targets = views[0]
        anchors = views[1]
        anchors_focal = torch.concat(views[2:], dim=0)

        targets_out = self.backbone(targets)
        targets_out = self.projection_head(targets_out)
        anchors_out = self.encode_masked(anchors)
        anchors_focal_out = self.encode_masked(anchors_focal)
        anchors_out = torch.cat([anchors_out, anchors_focal_out], dim=0)

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
        return self.anchor_projection_head(out)

    def configure_optimizers(self):
        params = [
            *list(self.anchor_backbone.parameters()),
            *list(self.anchor_projection_head.parameters()),
            self.prototypes,
        ]
        optimizer = torch.optim.AdamW(params, lr=self.lr,weight_decay=self.wd)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                         eta_min=0,
                                                         T_max=self.max_epochs
                                                         )
        
        return {'optimizer': optimizer,
                'lr_scheduler': scheduler
               }
    

# DINO model
class DINO(pl.LightningModule):
    def __init__(self,
                 image_size: int = 224,
                 patch_size: int = 16,
                 num_layers: int = 12,
                 num_heads: int = 6,
                 hiddin_dim: int = 192,
                 num_prototypes: int = 1024,
                 lr: float = 0.00001,
                 wd=0.0001,
                 max_epochs: int = 100):
        
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hiddin_dim
        self.num_prototypes = num_prototypes
        self.lr = lr
        self.wd = wd
        self.max_epochs = max_epochs
        
        self.backbone = MAEBackbone(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            hidden_dim=self.hidden_dim,
            mlp_dim=self.hidden_dim * 4,
        )


        self.student_backbone = self.backbone
        self.student_head = DINOProjectionHead(
            self.hidden_dim, 512, 256, 1024, freeze_last_layer=1)
        
        self.teacher_backbone = copy.deepcopy(self.backbone)
        self.teacher_head = DINOProjectionHead(self.hidden_dim, 512, 256, 1024)
        utils.deactivate_requires_grad(self.teacher_backbone)
        utils.deactivate_requires_grad(self.teacher_head)

        self.criterion = DINOLoss(output_dim=1024, warmup_teacher_temp_epochs=5)

    def forward(self, x):
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z

    def training_step(self, batch, batch_idx):
        momentum = cosine_schedule(self.current_epoch, 10, 0.996, 1)
        utils.update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        utils.update_momentum(self.student_head, self.teacher_head, m=momentum)
        views = batch[0]
        views = [view.to(self.device) for view in views]
        global_views = views[:2]
        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
        self.log("train_loss", loss, on_epoch= True,on_step=True , logger=True,prog_bar=True)
        return loss

    def on_after_backward(self):
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

    
    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(params=self.parameters(), 
                                   lr=self.learning_rate, 
                                   weight_decay=self.weight_decay
                                   )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                         eta_min=0,
                                                         T_max=self.max_epochs
                                                         )
        
        return {'optimizer': optimizer,
                'lr_scheduler': scheduler
               }



# Masked AutoEncoder Model
class MAE(pl.LightningModule):
    def __init__(self,
                 image_size: int = 224,
                 patch_size: int = 16,
                 num_layers: int = 12,
                 num_heads: int = 6,
                 hiddin_dim: int = 192,
                 mask_ratio: float = 0.6,
                 decoder_dim: int=512,
                 lr: float = 0.00001,
                 wd=0.0001,
                 max_epochs: int = 100):
        
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hiddin_dim
        self.mask_ratio = mask_ratio
        self.decoder_dim = decoder_dim
        self.lr = lr
        self.wd = wd
        self.max_epochs = max_epochs
  
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_dim))
        self.backbone = MAEBackbone(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            hidden_dim=self.hidden_dim,
            mlp_dim=self.hidden_dim*4)
        
        self.decoder = MAEDecoder(
            seq_length=self.backbone.seq_length,
            num_layers=8,
            num_heads=16,
            embed_input_dim=self.backbone.hidden_dim,
            hidden_dim=decoder_dim,
            mlp_dim=decoder_dim * 4,
            out_dim=self.backbone.patch_size**2 * 3,
            dropout=0,
            attention_dropout=0,
        )
        self.criterion = nn.MSELoss()

    def forward_encoder(self, images, idx_keep=None):
        return self.backbone.encode(images, idx_keep)

    def forward_decoder(self, x_encoded, idx_keep, idx_mask):
        # build decoder input
        batch_size = x_encoded.shape[0]
        x_decode = self.decoder.embed(x_encoded)
        x_masked = utils.repeat_token(
            self.mask_token, (batch_size, self.backbone.seq_length)
        )
        x_masked = utils.set_at_index(x_masked, idx_keep, x_decode.type_as(x_masked))

        # decoder forward pass
        x_decoded = self.decoder.decode(x_masked)

        # predict pixel values for masked tokens
        x_pred = utils.get_at_index(x_decoded, idx_mask)
        x_pred = self.decoder.predict(x_pred)
        return x_pred

    def training_step(self, batch, batch_idx):
        views = batch[0]
        images = views[0]  # views contains only a single view
        batch_size = images.shape[0]
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.backbone.seq_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )
        x_encoded = self.forward_encoder(images, idx_keep)
        x_pred = self.forward_decoder(x_encoded, idx_keep, idx_mask)

        # get image patches for masked tokens
        patches = utils.patchify(images, self.patch_size)
        # must adjust idx_mask for missing class token
        target = utils.get_at_index(patches, idx_mask - 1)

        loss = self.criterion(x_pred, target)
        self.log("train_loss", loss, on_epoch= True,on_step=True , logger=True,prog_bar=True)
        return loss

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(params=self.parameters(), 
                                       lr=self.learning_rate, 
                                       weight_decay=self.weight_decay
                                   )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                         eta_min=0,
                                                         T_max=self.max_epochs
                                                         )
        
        return {'optimizer': optimizer,
                'lr_scheduler': scheduler
               }

