from typing import Optional

from pathlib import Path
from omegaconf import OmegaConf
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from src.data.bottleneck_code.dataset import load_data

class CUBDataset(Dataset):
    def __init__(self, data: str, dtype: str = 'train'):
        
        self.dtype = dtype

        self.images = data['images']
        self.labels = data['labels']

        # Number of classes in dataset
        self.num_classes = self.labels.unique().__len__()

    def __len__(self):
        return self.labels.__len__()

    def __getitem__(self, item):
        return {
            "image": self.images[item, :], 
            "label": self.labels[item], 

            #TODO: add concepts as meta data
            #"concepts": self.concepts[item, :]
        }

def get_loaders(data_path: Path, batch_size: int = 128, shuffle: bool = True, num_workers: int = 1):
    # sourcery skip: dict-comprehension, inline-immediately-returned-variable
    print("INFO - Loading data...")
    # Load data
    data = torch.load(data_path.as_posix())
    normalization_vals = data['train']['normalization']

    # Construct dataset classes for splits
    trainData = CUBDataset(data=data['train'], dtype='train')
    print("INFO - training data loaded !")
    valData = CUBDataset(data=data['validation'], dtype='validation')
    print("INFO - validation data loaded !")
    testData = CUBDataset(data=data['test'], dtype='test')
    print("INFO - test data loaded !")
    
    # Create torch dataloaders
    loaders = {}
    for dtype, dataset_ in {'train': trainData, 'validation': valData, 'test': testData}.items():
        loaders[dtype] = torch.utils.data.DataLoader(
            dataset_,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
    return loaders, normalization_vals

def get_loaders_with_concepts(data_folder: Path, batch_size: int = 128):
    # sourcery skip: dict-comprehension, inline-immediately-returned-variable
    print("INFO - Loading data...")

    args = OmegaConf.create({
        'use_attr': True,
        'no_img': False,
        'batch_size': batch_size,
        'uncertain_labels': False,
        'image_dir': (data_folder / "raw/CUB_200_2011/CUB_200_2011/images").as_posix(),
        'n_class_attr': 2,
        'resampling': False,
    })

    # Specify paths
    train_data_path = (data_folder / "processed/CUB_200_2011/bottleneck/train.pkl").as_posix()
    val_data_path   = (data_folder / "processed/CUB_200_2011/bottleneck/val.pkl").as_posix()
    test_data_path  = (data_folder / "processed/CUB_200_2011/bottleneck/test.pkl").as_posix()

    train_loader = load_data(
        [train_data_path], 
        args.use_attr, 
        args.no_img, 
        args.batch_size, 
        args.uncertain_labels, 
        image_dir=args.image_dir,
        n_class_attr=args.n_class_attr, 
        resampling=args.resampling
    )
    print("INFO - training data loaded !")

    val_loader = load_data(
        [val_data_path], 
        args.use_attr, 
        args.no_img, 
        args.batch_size, 
        args.uncertain_labels, 
        image_dir=args.image_dir,
        n_class_attr=args.n_class_attr, 
        resampling=args.resampling
    )
    print("INFO - validation data loaded !")

    test_loader = load_data(
        [test_data_path], 
        args.use_attr, 
        args.no_img, 
        args.batch_size, 
        args.uncertain_labels, 
        image_dir=args.image_dir,
        n_class_attr=args.n_class_attr, 
        resampling=args.resampling
    )
    print("INFO - test data loaded !")

    loaders = {
        'train': train_loader,
        'validation': val_loader,
        'test': test_loader,
    }

    normalization_vals = {'mean': torch.tensor([0.5, 0.5, 0.5]), 'std': torch.tensor([2, 2, 2])}

    return loaders, normalization_vals