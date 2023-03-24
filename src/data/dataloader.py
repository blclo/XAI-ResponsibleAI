from typing import Optional

from pathlib import Path

import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

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