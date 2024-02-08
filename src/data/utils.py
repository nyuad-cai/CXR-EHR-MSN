from typing import Dict, List, Tuple, Union
import os
import torchvision.transforms as T
from PIL.Image import Image
import torch
import tqdm
import pandas as pd
from lightly.transforms.multi_view_transform import MultiViewTransform

from lightly.transforms.multi_view_transform import MultiViewTransform
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

class MSNTransform(MultiViewTransform):
    """Implements the transformations for MSN [0].

    Input to this transform:
        PIL Image or Tensor.

    Output of this transform:
        List of Tensor of length 2 * random_views + focal_views. (12 by default)

    Applies the following augmentations by default:
        - Random resized crop
        - Random horizontal flip
        - ImageNet normalization

    Generates a set of random and focal views for each input image. The generated output
    is (views, target, filenames) where views is list with the following entries:
    [random_views_0, random_views_1, ..., focal_views_0, focal_views_1, ...].

    - [0]: Masked Siamese Networks, 2022: https://arxiv.org/abs/2204.07141

    Attributes:
        random_size:
            Size of the random image views in pixels.
        focal_size:
            Size of the focal image views in pixels.
        random_views:
            Number of random views to generate.
        focal_views:
            Number of focal views to generate.
        random_crop_scale:
            Minimum and maximum size of the randomized crops for the relative to random_size.
        focal_crop_scale:
            Minimum and maximum size of the randomized crops relative to focal_size.
        hf_prob:
            Probability that horizontal flip is applied.
        vf_prob:
            Probability that vertical flip is applied.
        normalize:
            Dictionary with 'mean' and 'std' for torchvision.transforms.Normalize.
    """

    def __init__(
        self,
        random_size: int = 224,
        focal_size: int = 96,
        random_views: int = 2,
        focal_views: int = 10,
        affine_dgrees: int = 15,
        affine_scale: Tuple[float,float]= (.9, 1.1),
        affine_shear: int = 0,
        affine_translate: Tuple[float,float] = (0.1, 0.1),
        random_crop_scale: Tuple[float, float] = (0.3, 1.0),
        focal_crop_scale: Tuple[float, float] = (0.05, 0.3),
        hf_prob: float = 0.5,
        vf_prob: float = 0.0,
        normalize: Dict[str, List[float]] = IMAGENET_STAT,
    ):
        random_view_transform = MSNViewTransform(
            affine_dgrees=affine_dgrees,
            affine_scale=affine_scale,
            affine_shear=affine_shear,
            affine_translate=affine_translate,
            crop_size=random_size,
            crop_scale=random_crop_scale,
            hf_prob=hf_prob,
            vf_prob=vf_prob,
            normalize=normalize,
        )
        focal_view_transform = MSNViewTransform(
            affine_dgrees=affine_dgrees,
            affine_scale=affine_scale,
            affine_shear=affine_shear,
            affine_translate=affine_translate,
            crop_size=focal_size,
            crop_scale=focal_crop_scale,
            hf_prob=hf_prob,
            vf_prob=vf_prob,
            normalize=normalize,
        )
        transforms = [random_view_transform] * random_views
        transforms += [focal_view_transform] * focal_views
        super().__init__(transforms=transforms)


class MSNViewTransform:
    def __init__(
        self,
        affine_dgrees: int = 15,
        affine_scale: Tuple[float,float]= (.9, 1.1),
        affine_shear: int = 0,
        affine_translate: Tuple[float,float] = (0.1, 0.1),
        crop_size: int = 224,
        crop_scale: Tuple[float, float] = (0.3, 1.0),
        hf_prob: float = 0.5,
        vf_prob: float = 0.0,
        normalize: Dict[str, List[float]] = IMAGENET_STAT,
    ):

        transform = [
            T.RandomAffine(degrees=affine_dgrees, 
                          scale=affine_scale, 
                          shear=affine_shear, 
                          translate=affine_translate),
            T.RandomResizedCrop(size=crop_size, scale=crop_scale),
            T.RandomHorizontalFlip(p=hf_prob),
            T.RandomVerticalFlip(p=vf_prob),
            T.ToTensor(),
            T.Normalize(mean=normalize["mean"], std=normalize["std"]),
        ]

        self.transform = T.Compose(transform)

    def __call__(self, image: Union[torch.Tensor, Image]) -> torch.Tensor:
        """
        Applies the transforms to the input image.

        Args:
            image:
                The input image to apply the transforms to.

        Returns:
            The transformed image.

        """
        transformed = self.transform(image)
        return transformed
    


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