import os
import torch
import random

import pandas as pd
import torch.nn as nn
import torchvision.transforms as T

from PIL import Image
from typing import Tuple, Optional
from torch.utils.data import Dataset
from src.data.utils import preprocess



class CxrEhrDataset(Dataset):
    def __init__(self, 
                 data_dir: str,
                 transforms: nn.Module,
                 ) -> None:
      
        self.meta = pd.read_csv(data_dir)
        self.transforms = transforms
        self.ehr = self.meta.to_numpy()[:,1:].astype('float32')

        
    def __len__(self
                ) -> int:
        return len(self.meta)
    
    def __getitem__(self,
                    index: int
                    ) -> Tuple[torch.Tensor]:

        
        img = self.meta['path'][index]
        img = Image.open(fp=img).convert('RGB')
        img = self.transforms(img)
        ehr = torch.from_numpy(self.ehr[index])

        
        return img, ehr


class BaselinesDataset(Dataset):
    def __init__(self, 
                 data_dir: str, 
                 transform: nn.Module
                 ) -> None:
        self.transforms = transform
        self.data_dir = data_dir
        self.all_images = os.listdir(self.data_dir)
        for image in self.all_images:
            if image.startswith('._'):
                self.all_images.remove(image)
        
    def __len__(self) -> int:
        return len(self.all_images)
    
    def __getitem__(self,index: int
                    ) -> Tuple[torch.Tensor]:
        name = self.all_images[index]
        path = os.path.join(self.data_dir, name)
        img = Image.open(fp=path).convert('RGB')
        img = self.transform(img)
        return img, index, name



class MIMICCXR(Dataset):
    def __init__(self, 
                 paths: str,
                 data_dir: str, 
                 transform: Optional[T.Compose] = None, 
                 split: str = 'validate',
                 percentage:float = 1.0
                 ) -> None:
        self.data_dir = data_dir
        self.transform = transform
        self.filenames_to_path, \
        self.filenames_loaded, \
        self.filesnames_to_labels = preprocess(data_dir=self.data_dir,
                                               paths=paths,
                                               split=split
                                              )
        limit = (round(len(self.filenames_loaded) * percentage))
        self.filenames_loaded = random.sample(self.filenames_loaded,limit)
 
        
    def __getitem__(self, index):
        if isinstance(index, str):
            img = Image.open(self.filenames_to_path[index]).convert('RGB')
            labels = torch.tensor(self.filesnames_to_labels[index]).float()

            if self.transform is not None:
                img = self.transform(img)
            return img, labels
        
        filename = self.filenames_loaded[index]
        
        img = Image.open(self.filenames_to_path[filename]).convert('RGB')

        labels = torch.tensor(self.filesnames_to_labels[filename]).float()
        
            
        if self.transform is not None:
            img = self.transform(img)
        return img, labels

    def __len__(self):
        return len(self.filenames_loaded)
    
class CheXpertDataset(Dataset):
    def __init__(self, 
                 data_path: str,
                 transform: Optional[T.Compose] = None, 
                 ) -> None:
        
        self.transform = transform
        
        self.data = pd.read_csv(data_path)
        
    def __getitem__(self, index):
        img = Image.open(self.data['Path'][index]).convert('RGB')
        labels = self.data.iloc[index,1:].to_numpy().astype('float32')
        img = self.transform(img)
        return img, labels

    def __len__(self):
        return len(self.data)



class NIHDataset(Dataset):
    def __init__(self, root, data_path, transform=None):
        self.root = root
        self.df = pd.read_csv(data_path)
        self.transform = transform
        
        file = open(self.root)
        images = file.read().splitlines()
        
        ids = []
        
        for idx, path in enumerate(self.df['Image']):
            if path.split('/')[-1] in images:
                ids.append(idx)
        
        self.df = self.df.iloc[ids, :].reset_index(drop=True)
        self.images = self.df['Image'].values
        self.labels = self.df.iloc[:, 1:].values
        labels = list(map(lambda x: x.lower(), self.df.columns[1:]))
        self.classes = {v: k for k, v in enumerate(labels)}
        
    def __getitem__(self, item):
        img = Image.open(self.images[item]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            
        return img, torch.tensor(self.labels[item], dtype=torch.float32)
    
    def __len__(self):
        return len(self.df)