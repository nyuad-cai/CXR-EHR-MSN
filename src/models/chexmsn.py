import copy
import torch

import torch.nn as nn

from src.models.utils import deactivate_requires_grad, update_momentum, random_token_mask


class DenseBlock(nn.Module):
    def __init__(self,
                 in_features: int = 768,
                 out_features: int = 2048,
               bias: bool = False,
              ) -> None:

        super().__init__()

        self.dense_block = nn.Sequential(nn.Linear(in_features= in_features,
                                                   out_features= out_features,
                                                   bias=bias
                                                   ),
                                         nn.LayerNorm(normalized_shape= out_features),
                                         nn.GELU()
                                        )
    def forward(self,
                x: torch.Tensor
                ) -> torch.tensor:
        x = self.dense_block(x)
        return x
    


class ProjectionHead(nn.Module):
    def __init__(self,
                 in_features: int = 512,
                 hidden_features: int = 2048,
                 out_features: int = 512,
                 bias : bool = False
                 ) -> None:

        super().__init__()

        self.projection_head = nn.Sequential(DenseBlock(in_features= in_features,
                                                    out_features= hidden_features,
                                                    bias= bias
                                                    ),
                                         DenseBlock(in_features= hidden_features,
                                                    out_features= hidden_features,
                                                    bias= bias
                                                    ),
                                         nn.Linear(in_features= hidden_features,
                                                   out_features= out_features,
                                                   bias= bias
                                                   )
                                         )

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        x = self.projection_head(x)
        return x
    


class MSN(nn.Module):
    def __init__(self,
               backbone: nn.Module,
               masking_ratio: float = 0.15,
               ema_p: float = 0.996
              ) -> None:
        super().__init__()

        self.masking_ratio = masking_ratio
        self.ema_p = ema_p

        self.anchor_backbone = backbone
        self.anchor_projection_head = ProjectionHead()

        self.target_backbone = copy.deepcopy(self.anchor_backbone)
        self.target_projection_head = copy.deepcopy(self.anchor_projection_head)

        deactivate_requires_grad(self.target_backbone)
        deactivate_requires_grad(self.target_projection_head)



    def forward(self,
                views:list[torch.tensor],
                focal: bool = True
                ) -> tuple[torch.Tensor]:

        update_momentum(model= self.anchor_backbone,
                        model_ema= self.target_backbone,
                        m = self.ema_p,
                       )
        update_momentum(model= self.anchor_projection_head,
                        model_ema= self.target_projection_head,
                        m = self.ema_p,
                       )
        projections = self._forward_all(batch=views,focal=focal)
        

        return projections

    def _target_forward(self,
                        view: torch.tensor
                        ) -> torch.Tensor:

        target_encodings = self.target_backbone(x= view,
                                                branch='target'
                                                )
        target_projections = self.target_projection_head(x= target_encodings)

        return target_projections


    def _anchor_forward(self,
                        view: torch.tensor
                        ) -> torch.Tensor:

        batch_size, _, _, width = view.shape
        seq_length = (width // self.anchor_backbone.patch_size) ** 2
        idx_keep, idx_mask = random_token_mask(size= (view.shape[0],seq_length),
                                               mask_ratio= self.masking_ratio
                                          )

        anchor_encodings = self.anchor_backbone(x= view,
                                                branch= 'anchor',
                                                idx_keep= idx_keep)
        anchor_projections = self.anchor_projection_head(x= anchor_encodings)

        return anchor_projections
    
    
    def _forward_all(self,
                     batch: list,
                     focal: bool = True
                     ) -> torch.tensor:
        target_view = batch[0]
        anchor_view = batch[1]
#        focal_views = torch.concat(batch[2:],dim=0)
        
        target_projections = self._target_forward(target_view)
        
        anchor_projections_sim = self._anchor_forward(anchor_view)
#         if focal:
        #anchor_focal_projections_sim = self._anchor_forward(focal_views)
#             similarity_projections = torch.cat([anchor_projections_sim,
#                                                 anchor_focal_projections_sim],
#                                                 dim= 0)
        
        anchor_projections_age = self._anchor_forward(anchor_view)
#         if focal:
#        anchor_focal_projections_age =  self._anchor_forward(focal_views)
#             age_projections = torch.cat([anchor_projections_age,
#                                                 anchor_focal_projections_age],
#                                                 dim= 0)
        
        
        anchor_projections_gender = self._anchor_forward(anchor_view)
#         if focal:
#        anchor_focal_projections_gender = self._anchor_forward(focal_views)  
#             gender_projections = torch.cat([anchor_projections_sim,
#                                             anchor_focal_projections_sim],
#                                             dim= 0)            
        
        anchor_projections = torch.stack((anchor_projections_sim,
                                          anchor_projections_age,
                                          anchor_projections_gender
                                          ),
                                         dim= 1
                                         )
       
        

        return (target_projections,
                anchor_projections,
               )
