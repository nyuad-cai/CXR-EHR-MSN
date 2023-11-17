

import os
import torch
import torch.nn as nn 
import pytorch_lightning as pl

from typing import Tuple, Optional
from torch import Tensor
from PIL import Image
from torch.utils.data import Dataset


class ChexMSNDataset(nn.Module):
    pass


class PretrainDataset(Dataset):
    def __init__(self, 
                 data_dir: str, 
                 mode:str
                 ) -> None:
      
        self.data_dir = data_dir
        self.mode = mode
        self.all_images = os.listdir(self.data_dir)
        for image in self.all_images:
            if image.startswith('._'):
                self.all_images.remove(image)
        
    def __len__(self
                ) -> int:
        return len(self.all_images)
    
    def __getitem__(self,
                    index: int
                    ) -> Tuple[Tensor]:

        
        name = self.all_images[index]
        path = os.path.join(self.data_dir, name)
        img = Image.open(fp=path).convert('RGB')
        if self.mode == 'dino':
            pass
        #     img = transform1(img)
        # elif self.mode == 'simsiam':
        #     img = transform2(img)


        return img, index, name