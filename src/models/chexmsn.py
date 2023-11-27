import copy
import torch

import torch.nn as nn
import pytorch_lightning as pl
import lightly.models.utils as utils
from typing import List
from src.models.utils import deactivate_requires_grad, update_momentum, random_token_mask
from torch.utils.data import DataLoader
import torchvision
from lightly.utils.benchmarking import BenchmarkModule
import torch.distributed as dist
from lightly.loss import MSNLoss
from lightly.models.modules.heads import MSNProjectionHead
from lightly.models.modules.masked_autoencoder import MAEBackbone

from src.models.visiontransformer import VisionTransformer

class DenseBlock(nn.Module):
    def __init__(self,
                 in_features: int = 768,
                 out_features: int = 2048,
               bias: bool = False,
              ) -> None:

        super().__init__()

        self.dense_block = nn.Sequential(nn.Linear(in_features= in_features,
                                                   out_features= out_features,
                                               bias=bias),
                                         nn.LayerNorm(normalized_shape= out_features),
                                         nn.GELU()
                                        )
    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        x = self.dense_block(x)
        return x
    

class ProjectionHead(nn.Module):
    def __init__(self,
                 in_features: int = 768,
                 hidden_features: int = 2048,
                 out_features: int = 256,
                 bias : bool = False
                 ) -> None:

        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.projection_head = nn.Sequential(DenseBlock(in_features= self.in_features,
                                                    out_features= self.hidden_features,
                                                    bias= bias
                                                    ),
                                         DenseBlock(in_features= self.hidden_features,
                                                    out_features= self.hidden_features,
                                                    bias= bias
                                                    ),
                                         nn.Linear(in_features= self.hidden_features,
                                                   out_features= self.out_features,
                                                   bias= bias
                                                   )
                                         )

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        x = self.projection_head(x)
        return x
    
class ChexMSN(nn.Module):
    def __init__(self,
               backbone: nn.Module,
               projection_head: nn.Module,
               masking_ratio: float = 0.15,
               ema_p: float = 0.996,
               focal: bool = True
              ) -> None:
        super().__init__()

        self.masking_ratio = masking_ratio
        self.ema_p = ema_p
        self.focal = focal

        self.backbone = backbone
        self.projection_head = projection_head

        self.target_backbone = copy.deepcopy(self.backbone)
        self.target_projection_head = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.target_backbone)
        deactivate_requires_grad(self.target_projection_head)



    def forward(self,
                views:list[torch.Tensor],
                focal: bool = True
                ) -> tuple[torch.Tensor]:

        update_momentum(model= self.backbone,
                        model_ema= self.target_backbone,
                        m = self.ema_p,
                       )
        update_momentum(model= self.projection_head,
                        model_ema= self.target_projection_head,
                        m = self.ema_p,
                       )
        projections = self._forward_all(batch=views,focal=self.focal)
        

        return projections

    def _target_forward(self,
                        view: torch.Tensor
                        ) -> torch.Tensor:

        target_encodings = self.target_backbone(x= view,
                                                branch='target')
        target_projections = self.target_projection_head(x= target_encodings)

        return target_projections


    def _anchor_forward(self,
                        view: torch.Tensor
                        ) -> torch.Tensor:

        batch_size, _, _, width = view.shape
        seq_length = (width // self.backbone.patch_size) ** 2
        idx_keep, idx_mask = random_token_mask(size= (view.shape[0],seq_length),
                                               mask_ratio= self.masking_ratio,
                                               device=torch.device("cuda"))

        anchor_encodings = self.backbone(x= view,
                                         branch= 'anchor',
                                         idx_keep= idx_keep)
        anchor_projections = self.projection_head(x= anchor_encodings)

        return anchor_projections
    
    
    def _forward_all(self,
                     batch: list,
                     focal: bool = True
                     ) -> torch.Tensor:
        
        target_view = batch[0][0].to('cuda')
        anchor_view_sim = batch[0][1].to('cuda')
        focal_views_sim = torch.concat(batch[0][2:],dim=0).to('cuda')
        anchor_view_age = batch[1][1].to('cuda')
        focal_views_age = torch.concat(batch[1][2:],dim=0).to('cuda')
        anchor_view_gender = batch[2][1].to('cuda')
        focal_views_gender = torch.concat(batch[2][2:],dim=0).to('cuda')
        
        
        target_projections = self._target_forward(target_view)
        
        anchor_projections_sim = self._anchor_forward(anchor_view_sim)
        if focal:
            anchor_focal_projections_sim = self._anchor_forward(focal_views_sim)
            similarity_projections = self._arrange_tokens(anchor_projections_sim,
                                                          anchor_focal_projections_sim,
                                                          num_focal = 10)
                                                          
        
        anchor_projections_age = self._anchor_forward(anchor_view_age)
        if focal:
            anchor_focal_projections_age =  self._anchor_forward(focal_views_age)
            age_projections = self._arrange_tokens(anchor_projections_age,
                                                   anchor_focal_projections_age,
                                                   num_focal = 10)
                                                    
        
        
        anchor_projections_gender = self._anchor_forward(anchor_view_gender)
        if focal:
            anchor_focal_projections_gender = self._anchor_forward(focal_views_gender)  
            gender_projections = self._arrange_tokens(anchor_projections_gender,
                                                      anchor_focal_projections_gender,
                                                      num_focal = 10)
                                                          
        if focal:
            anchor_projections = torch.stack((similarity_projections,
                                              age_projections,
                                              gender_projections
                                              ),
                                         dim= 0)
        else:
            anchor_projections = torch.stack((anchor_projections_sim,
                                              anchor_projections_age,
                                              anchor_projections_gender
                                              ),
                                             dim= 1)       

        return (anchor_projections,
                target_projections)
    
    def _arrange_tokens(self,
                        tensor1: torch.Tensor,
                        tensor2:torch.Tensor,
                        num_focal: int = 10
                        ) ->torch.Tensor:

        a = torch.stack(torch.split(tensor1,1),0)
        b = torch.stack(torch.split(tensor2,num_focal),0)
        c = torch.cat((a.expand(-1,num_focal,-1),b),dim=1)[:,num_focal-1:]
        arranged_tokens = torch.cat(c.split(1),1).squeeze(0)
        return arranged_tokens
    


class ChexMSNModel(pl.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 criterion: nn.Module,
                 num_prototypes: int = 1024,
                 learning_rate: float =  1e-3,
                 weight_decay: float = 0.0,
                 max_epochs: int = 50,
                 focal: bool = True

                ) -> None:
        super().__init__()

        self.model = model
        self.criterion = criterion
        self.num_prototypes = num_prototypes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.focal = focal
        self.in_prototypes = self.model.projection_head.out_features
        self.prototypes = nn.ModuleList([nn.Linear(in_features=self.in_prototypes,out_features=num_prototypes,bias=False),
                                         nn.Linear(in_features=self.in_prototypes,out_features=num_prototypes,bias=False),
                                         nn.Linear(in_features=self.in_prototypes,out_features=num_prototypes,bias=False)])
    
    
    def training_step(self, 
                      batch: List[torch.Tensor], 
                      batch_idx: int
                     ) -> float:
        

        anchors, target = self.model(batch) 
        loss = self.criterion(anchors,target,self.prototypes,focal=self.focal)
        self.log("train_loss", loss, on_epoch= True,on_step=True, logger=True,)
        return loss


    def configure_optimizers(self):

        params = [
            *list(self.model.backbone.parameters()),
            *list(self.model.projection_head.parameters()),
            self.prototypes[0].weight,
            self.prototypes[1].weight,
            self.prototypes[2].weight,
        ]
        
        optimizer = torch.optim.AdamW(params, 
                                      lr=self.learning_rate,
                                      weight_decay=self.weight_decay)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                               eta_min=0.00001,
                                                               T_max=self.max_epochs)
        return {'optimizer': optimizer,
               'lr_scheduler': scheduler
               }
    

class MSN(BenchmarkModule):
    pass
    # def __init__(self,
    #            dataloader_kNN: DataLoader, 
    #            num_classes: int = 2,
    #            knn_k: int = 5,
    #            knn_t: float = 0.1,
    #            mask_ratio: float = 0.15,
    #            lr : float = 0.1,
    #            prototypes_num: int = 1024,
    #            weight_decay: float = 0.0,
    #            max_epochs: int = 100
    #            ) -> None:
    #     super().__init__(dataloader_kNN, num_classes,knn_k,knn_t)
    #     #self.save_hyperparameters() 
    #     self.weight_decay = weight_decay
    #     self.mask_ratio = mask_ratio
    #     self.lr = lr
    #     self.prototypes_num = prototypes_num
    #     self.max_epochs = max_epochs
       
    #     vit = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT)
    #     self.backbone = MAEBackbone.from_vit(vit)
    #     self.projection_head = MSNProjectionHead(768)

    #     self.anchor_backbone = copy.deepcopy(self.backbone)
    #     self.anchor_projection_head = copy.deepcopy(self.projection_head)

    #     deactivate_requires_grad(self.backbone)
    #     deactivate_requires_grad(self.projection_head)

    #     self.prototypes = nn.Linear(256, self.prototypes_num, bias=False).weight
    #     self.criterion = MSNLoss()

    # def training_step(self, batch, batch_idx):
    #     utils.update_momentum(self.anchor_backbone, self.backbone, 0.996)
    #     utils.update_momentum(self.anchor_projection_head, self.projection_head, 0.996)

    #     views, _, _ = batch
    #     views = [view.to(self.device, non_blocking=True) for view in views]
    #     targets = views[0]
    #     anchors = views[1]
    #     anchors_focal = torch.concat(views[2:], dim=0)

    #     targets_out = self.backbone(targets)
    #     targets_out = self.projection_head(targets_out)
    #     anchors_out = self.encode_masked(anchors)
    #     anchors_focal_out = self.encode_masked(anchors_focal)
    #     anchors_out = torch.cat([anchors_out, anchors_focal_out], dim=0)

    #     loss = self.criterion(anchors_out, targets_out, self.prototypes.data)
    #     self.log("train_loss", loss, on_epoch= True,on_step=True , logger=True, prog_bar=True)
    #     return loss
    
        

    # def encode_masked(self, anchors):
    #     batch_size, _, _, width = anchors.shape
    #     seq_length = (width // self.anchor_backbone.patch_size) ** 2
    #     idx_keep, _ = utils.random_token_mask(size=(batch_size, seq_length),
    #                                           mask_ratio=self.mask_ratio,
    #                                           device=self.device,
    #                                           )
    #     out = self.anchor_backbone(anchors, idx_keep)
    #     return self.anchor_projection_head(out)
    
    

    # def configure_optimizers(self):
    #     params = [
    #         *list(self.anchor_backbone.parameters()),
    #         *list(self.anchor_projection_head.parameters()),
    #         self.prototypes,
    #     ]

    #     optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr,weight_decay=self.weight_decay)
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
    #                                                            eta_min=0.00001,
    #                                                            T_max=self.max_epochs
    #                                                         )
    #     return {'optimizer': optimizer,
    #            'lr_scheduler': scheduler
    #            }



class MSN1(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # ViT small configuration (ViT-S/16)
        self.mask_ratio = 0.15
        # self.backbone = MAEBackbone(
        #     image_size=224,
        #     patch_size=16,
        #     num_layers=12,
        #     num_heads=6,
        #     hidden_dim=384,
        #     mlp_dim=384 * 4,
        # )
        # # or use a torchvision ViT backbone:
        vit = VisionTransformer(num_heads=6,hidden_dim=768,mlp_dim=768*4,num_cls_tokens=1)
        self.backbone = vit#MAEBackbone.from_vit(vit)
        self.projection_head = MSNProjectionHead(768)

        self.anchor_backbone = copy.deepcopy(self.backbone)
        self.anchor_projection_head = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone)
        deactivate_requires_grad(self.projection_head)

        self.prototypes = nn.Linear(256, 1024, bias=False).weight
        self.criterion = MSNLoss()

    def training_step(self, batch, batch_idx):
        update_momentum(self.anchor_backbone, self.backbone, 0.996)
        update_momentum(self.anchor_projection_head, self.projection_head, 0.996)

        views = batch[0]
        views = [view.to(self.device, non_blocking=True) for view in views]
        targets = views[0]
        anchors = views[1]
        anchors_focal = torch.concat(views[2:], dim=0)

        targets_out = self.backbone(targets,branch='anchor')
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
        out = self.anchor_backbone(anchors,branch='anchor', idx_keep=idx_keep)
        return self.anchor_projection_head(out)

    def configure_optimizers(self):
        params = [
            *list(self.anchor_backbone.parameters()),
            *list(self.anchor_projection_head.parameters()),
            self.prototypes,
        ]
        optim = torch.optim.AdamW(params, lr=0.0001,weight_decay=0.001)
        return optim