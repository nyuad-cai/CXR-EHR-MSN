import torch
import random
import pandas as pd
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from typing import Tuple, Optional
from torch.utils.data import Dataset
from src.data.utils import preprocess


class ChexMSNDataset(Dataset):
    def __init__(self, 
                 data_dir: str,
                 transforms: nn.Module,
                 same = True
                 ) -> None:
      
        self.meta = pd.read_csv(data_dir)
        self.all_images = list(self.meta.path)
        self.transform = transforms
        self.same = same
        
    def __len__(self
                ) -> int:
        return len(self.all_images)
    
    def __getitem__(self,
                    index: int
                    ) -> Tuple[torch.Tensor]:

        
        target_path = self.all_images[index]
        image_id = target_path.split('/')[-1][:-4]
        img_age_path, img_gender_path = self._retrieve_anchors(image_id=image_id,
                                                               meta = self.meta,
                                                               same=self.same)

        img_target = Image.open(fp=target_path).convert('RGB')
        img_target = self.transform(img_target)
        
        img_age = Image.open(fp=img_age_path).convert('RGB')
        img_age = self.transform(img_age)

        img_gender = Image.open(fp=img_gender_path).convert('RGB')
        img_gender = self.transform(img_gender)

        return (img_target,img_age,img_gender)
    
    
    def _retrieve_anchors(self,
                          image_id: str,
                          meta: pd.DataFrame,
                          same: bool = False) -> Tuple[str]:
        record = meta[meta.dicom_id == image_id]
    
        subject_id = list(record.subject_id)[0]
        age_groub =list(record.ageR5)[0] 
        gender = list(record.gender)[0]
    
        group = meta[meta.ageR5 == age_groub]
    
        if same:
            candidate_anchors = group[group.gender == gender]
            candidate_anchors = candidate_anchors[candidate_anchors.subject_id != subject_id]
            images= list(candidate_anchors.path)
            sampled_images = random.sample(images,k=2)
            image_age, image_gender = sampled_images[0],sampled_images[1]
            return image_age, image_gender
        else:
            candidate_anchors = group
            candidate_anchors = candidate_anchors[candidate_anchors.subject_id != subject_id]
            images= list(candidate_anchors.path)
            image_age = random.sample(images,k=1)[0]
            candidate_anchors = candidate_anchors[candidate_anchors.gender == gender]
            images= list(candidate_anchors.path)
            image_gender = random.sample(images,k=1)[0]
            return image_age, image_gender
        

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
        self.filenames_loaded = self.filenames_loaded[0:limit]
 
        
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