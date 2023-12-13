import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import pandas as pd
from torch import Tensor
from lightly.models import utils
from typing import List
from sklearn.metrics import roc_auc_score, average_precision_score
from src.models.utils import get_model_performance

class EvaluationModel(pl.LightningModule):
    def __init__(self,
                backbone:nn.Module,
                learning_rate: float =  1e-3,
                weight_decay: float = 0.0,
                output_dim: int = 14,
                freeze: bool = False,
                max_epochs: int = 50,
                mask_ratio: float = 0.15,
                ) -> None:
        super().__init__()

        # self.save_hyperparameters() 
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.output_dim = output_dim
        self.mask_ratio = mask_ratio
        self.backbone = backbone
        
        self.train_step_preds = []
        self.train_step_label = []
        
        self.val_step_preds = []
        self.val_step_label = []
        
        self.test_step_preds = []
        self.test_step_label = []
        
     
        if freeze:
            utils.deactivate_requires_grad(self.backbone)
        else:
            utils.activate_requires_grad(self.backbone)
               
    def forward(self,
                x: Tensor
               ) -> Tensor:
        x = self.backbone(x)
        x = torch.sigmoid(x)
        return x
    
    

    def training_step(self, 
                      batch: List[Tensor], 
                      batch_idx: int
                     ) -> float:
        
        input, label = batch
        prediction = self.forward(input)
        
        self.train_step_label.append(label)
        self.train_step_preds.append(prediction)

        loss = nn.BCELoss()(prediction, label)            
        self.log("train_loss", loss, on_epoch= True,on_step=False , logger=True, prog_bar=True)
        
        return {'loss':loss,
                'pred':prediction,
                'label':label}

    def on_train_epoch_end(self) -> None:


        
        y = torch.cat(self.train_step_label).detach().cpu()
        pred = torch.cat(self.train_step_preds).detach().cpu()

        auroc = np.round(roc_auc_score(y, pred), 4)
        auprc = np.round(average_precision_score(y, pred), 4)   
        self.log('train_auroc',auroc, on_epoch=True, on_step=False,logger=True, prog_bar=True)
        self.log('train_auprc',auprc, on_epoch=True, on_step=False,logger=True, prog_bar=True)      
        self.train_step_label.clear()
        self.train_step_preds.clear()
        
    def validation_step (self, 
                      batch: List[Tensor], 
                      batch_idx: int
                     ) -> float:
        
        input,label = batch
        prediction = self.forward(input) 
        
        self.val_step_label.append(label)
        self.val_step_preds.append(prediction)

        loss = self._bce_loss(prediction, label,mode='val')       
        self.log("val_loss", loss, on_epoch= True,on_step=False,logger=True, prog_bar=True)

        return {'loss':loss,
                'pred':prediction,
                'label':label}

    def on_validation_epoch_end(self,*arg, **kwargs) -> None:
        
        y = torch.cat(self.val_step_label).detach().cpu()
        pred = torch.cat(self.val_step_preds).detach().cpu()

        auroc = np.round(roc_auc_score(y, pred), 4)
        auprc = np.round(average_precision_score(y, pred), 4)   
        self.log('val_auroc',auroc, on_epoch=True, on_step=False, logger=True, prog_bar=True)
        self.log('val_auprc',auprc, on_epoch=True, on_step=False, logger=True, prog_bar=True)    
        self.val_step_label.clear()
        self.val_step_preds.clear()
        
    def test_step(self, 
                  batch: List[Tensor], 
                  batch_idx: int
                 ) -> float:
        input, label = batch
        prediction = self.forward(input)
        
        self.test_step_label.append(label)
        self.test_step_preds.append(prediction)
        
        loss = self._bce_loss(prediction, label,mode='test')
        self.log("test_loss", loss, on_epoch= True,on_step=False , logger=True, prog_bar=True)

        
        return {'loss':loss,
                'pred':prediction,
                'label':label}

    def on_test_epoch_end(self,*arg, **kwargs) -> None:
        y = torch.cat(self.test_step_label).detach().cpu()
        pred = torch.cat(self.test_step_preds).detach().cpu()


        auroc = np.round(roc_auc_score(y, pred), 4)
        auprc = np.round(average_precision_score(y, pred), 4)   
        self.log('test_auroc',auroc, on_epoch=True, on_step=False, logger=True)
        self.log('test_auprc',auprc, on_epoch=True, on_step=False, logger=True) 
        
        df = pd.DataFrame()
        df['y_truth'] = y.tolist()
        df['y_pred'] = pred.tolist()
        get_model_performance(df)            
          
        self.test_step_label.clear()
        self.test_step_preds.clear()

    def configure_optimizers(self):

        optimizer = optim.Adam(params=self.parameters(), 
                                   lr=self.learning_rate, 
                                   weight_decay=self.weight_decay
                                   )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                         eta_min=0,
                                                         T_max=self.max_epochs
                                                         )
        
        return {'optimizer': optimizer,
                'lr_scheduler': scheduler
               }
    

    
    def _bce_loss(self, preds, y,mode='train'):
        loss = nn.BCELoss()(preds, y)
        if torch.is_tensor(y):
            y = y.detach().cpu().numpy()
        return loss
