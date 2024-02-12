
import os
import tqdm
import torch

import pandas as pd
import torchvision.transforms as T


from typing import List, Tuple



IMAGENET_STAT = {"mean":torch.tensor([0.4884, 0.4550, 0.4171]),
                 "std":torch.tensor([0.2596, 0.2530, 0.2556])}



train_transforms = T.Compose([T.Resize(256),
                              T.RandomHorizontalFlip(),
                              T.RandomAffine(degrees=45, scale=(.85, 1.15), shear=0, translate=(0.15, 0.15)),
                              T.CenterCrop(224),
                              T.ToTensor(),
                              T.Normalize(mean=IMAGENET_STAT["mean"],
                                          std=IMAGENET_STAT["std"])                                                                   
                            ])


val_test_transforms = T.Compose([T.Resize(256),
                                 T.CenterCrop(224),
                                 T.ToTensor(),
                                 T.Normalize(mean=IMAGENET_STAT["mean"],
                                             std=IMAGENET_STAT["std"])                                                                
                            ])


def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data in tqdm(dataloader):
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    
    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


def preprocess(data_dir:str,
               paths: list,
               split:str,
               ) -> Tuple[List]:
    CLASSES  = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
                'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
                'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
                'Pneumonia', 'Pneumothorax', 'Support Devices']
        
    filenames_to_path = {path.split('/')[-1].split('.')[0]: path for path in paths}
    metadata = pd.read_csv(os.path.join(data_dir,'mimic-cxr-2.0.0-metadata.csv'))
    labels = pd.read_csv(os.path.join(data_dir,'mimic-cxr-2.0.0-chexpert.csv'))
    labels[CLASSES] = labels[CLASSES].fillna(0)
    labels = labels.replace(-1.0, 0.0)
    splits = pd.read_csv(os.path.join(data_dir,'mimic-cxr-ehr-split.csv'))
    metadata_with_labels = metadata.merge(labels[CLASSES+['study_id'] ], how='inner', on='study_id')
    filesnames_to_labels = dict(zip(metadata_with_labels['dicom_id'].values, metadata_with_labels[CLASSES].values))
    filenames_loaded = splits.loc[splits.split==split]['dicom_id'].values
    
    filenames_loaded = [filename  for filename in filenames_loaded if filename in filesnames_to_labels]

    return filenames_to_path, filenames_loaded, filesnames_to_labels