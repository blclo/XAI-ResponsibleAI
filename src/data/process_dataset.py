from typing import Optional

import os
from pathlib2 import Path
import os

from datetime import datetime
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torchvision.transforms as T

def get_meta(data_path):

    with open(data_path / 'classes.txt', 'r') as file:
        lines = file.readlines()

        info_ = [line.split() for line in lines]

    idx2label = {int(class_idx)-1: label for class_idx, label in info_}
    label2idx = {label: int(class_idx)-1 for class_idx, label in info_}
    return idx2label, label2idx

def load_txt_files(data_path):
    # Read the file
    with open(data_path / 'image_class_labels.txt', 'r') as file:
        lines = file.readlines()

        info_ = [line.split() for line in lines]
        image_idxs, labels = list(zip(*info_))

    # Read the file
    with open(data_path / 'train_test_split.txt', 'r') as file:
        lines = file.readlines()

        info_ = [line.split() for line in lines]
        image_idxs, train_test_split = list(zip(*info_))

    with open(data_path / 'images.txt', 'r') as file:
        lines = file.readlines()

        info_ = [line.split() for line in lines]
        image_idxs, image_paths = list(zip(*info_))


    return pd.DataFrame({
        'index': np.array(image_idxs).astype(int), 
        'label': np.array(labels).astype(int), 
        'is_training_image': np.array(train_test_split).astype(int),
        'image_path': image_paths,
    })

def plot_example_images(images, labels, N, target_size, idx2label, type='TRAIN', output_path: Optional[str] = None):
    example_idxs = np.random.choice(np.arange(images.__len__()), N)

    fig, axs = plt.subplots(2, 5, figsize=(10, 5))
    for i in range(5):
        axs[0, i].imshow(images[example_idxs[i], :, :, :].squeeze().permute(1, 2, 0))
        axs[0, i].set_title(idx2label[labels[example_idxs[i]]])

        axs[1, i].imshow(images[example_idxs[5+i], :, :, :].squeeze().permute(1, 2, 0))
        axs[1, i].set_title(idx2label[labels[example_idxs[5+i]]])
    fig.suptitle(f'VISUAL INSPECTION OF {type}LOADER')
    plt.tight_layout()

    if output_path is not None:
        figures_path = output_path.parent.parent.parent / 'figures' / 'data_processing'
        os.makedirs(figures_path, exist_ok=True)
        plt.savefig(figures_path / f'{target_size}x{target_size}_{type}example-images')
    else:
        plt.show()
    
def create_pickle_files(data_path: Path, output_filename: str, output_path: Path, target_size: int = 256, show_examples: bool = True, seed: int = 0):
    print("")

    output_path = Path(output_path).resolve()

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(42)

    # Define preprocess pipeline
    preprocess = T.Compose(
        [   
            T.Resize(target_size),
            T.CenterCrop(target_size),
            T.ToTensor(),
        ]
    )
    
    # Load meta-data
    idx2label, label2idx = get_meta(data_path)
    
    # Load information from txt-files (including train-test-split)
    ## df.sample(frac=1.0) is for shuffling the dataframes
    df = load_txt_files(data_path)
    df_train = df.query('is_training_image == 1').sample(frac=1.0)
    df_valtest = df.query('is_training_image == 0').sample(frac=1.0)
    
    # Split test set into validation and test set
    valsize = int(df_valtest.shape[0] * 0.4)
    testsize = df_valtest.shape[0] - valsize

    df_val = df_valtest[:valsize]
    df_test = df_valtest[valsize:]

    # Prepare for loading training data
    train_images, train_labels, train_idxs, train_img_paths = [], [], [], []
    # Loop through training images
    for idx, row in tqdm(df_train.iterrows(), total=len(df_train), desc='Processing training data...'):
        # Get path
        img = Image.open((data_path / 'images' / row['image_path']).as_posix())
        # Preprocess image
        x = preprocess(img)

        if x.shape[0] == 3: # Only store color images
            train_idxs.append(idx)
            train_labels.append(row['label'])
            train_images.append(preprocess(img))
            train_img_paths.append(row['image_path'])

    # Prepare for loading validation data
    val_images, val_labels, val_idxs, val_img_paths = [], [], [], []
    # Loop through validation images
    for idx, row in tqdm(df_val.iterrows(), total=len(df_val), desc='Processing validation data...'):
        # Get path
        img = Image.open((data_path / 'images' / row['image_path']).as_posix())
        # Preprocess image
        x = preprocess(img)

        if x.shape[0] == 3: # Only store color images
            val_idxs.append(idx)
            val_labels.append(row['label'])
            val_images.append(preprocess(img))
            val_img_paths.append(row['image_path'])

    # Prepare for loading validation data
    test_images, test_labels, test_idxs, test_img_paths = [], [], [], []
    # Loop through test images
    for idx, row in tqdm(df_test.iterrows(), total=len(df_test), desc='Processing test data...'):
        # Get path
        img = Image.open((data_path / 'images' / row['image_path']).as_posix())
        # Preprocess image
        x = preprocess(img)

        if x.shape[0] == 3: # Only store color images
            test_idxs.append(idx)
            test_labels.append(row['label'])
            test_images.append(preprocess(img))
            test_img_paths.append(row['image_path'])

    # Stack images and labels
    train_images = torch.stack(train_images)
    val_images = torch.stack(val_images)
    test_images = torch.stack(test_images)

    train_labels = torch.tensor(train_labels) - 1
    val_labels = torch.tensor(val_labels) - 1
    test_labels = torch.tensor(test_labels) - 1
    
    print(f"\n{'-'*50}\nDATASET SPLITS\n{'-'*50}")
    print(f"INFO - Training set: {train_images.__len__() / df.__len__()}")
    print(f"INFO - Validation set: {val_images.__len__() / df.__len__()}")
    print(f"INFO - Test set: {test_images.__len__() / df.__len__()}")
    print(f"{'-'*50}")

    # Do a visual inspection to see if the loading of files works
    if show_examples:
        N = 10
        plot_example_images(train_images, train_labels.numpy(), N, target_size, idx2label, type='TRAIN-', output_path=output_path)
        plot_example_images(val_images, val_labels.numpy(), N, target_size, idx2label, type='VALIDATION-', output_path=output_path)
        plot_example_images(test_images, test_labels.numpy(), N, target_size, idx2label, type='TEST-', output_path=output_path)


    ### Normalization ### 
    # Compute mean and standard deviation of training set for normalization

    mean_train, std_train = train_images.mean(axis=[0, 2, 3]), train_images.std(axis=[0, 2, 3])
    mean_val, std_val = val_images.mean(axis=[0, 2, 3]), val_images.std(axis=[0, 2, 3])
    mean_test, std_test = test_images.mean(axis=[0, 2, 3]), test_images.std(axis=[0, 2, 3])

    print("")
    print(f"INFO - Training images (before normalization): \n\tMean: {mean_train.numpy()} | Std: {std_train.numpy()}")
    print(f"INFO - Validation images (before normalization): \n\tMean: {mean_val.numpy()} | Std: {std_val.numpy()}")
    print(f"INFO - Test images (before normalization): \n\tMean: {mean_test.numpy()} | Std: {std_test.numpy()}")
    print("")

    # Define normalization pipeline
    normalize = T.Normalize(mean=list(mean_train.numpy()), std=list(std_train.numpy()))

    # Normalize wrt. training stats
    train_images = torch.stack([normalize(img) for img in tqdm(train_images, desc='Normalizing training images...')])
    val_images = torch.stack([normalize(img) for img in tqdm(val_images, desc='Normalizing validation images...')])
    test_images = torch.stack([normalize(img) for img in tqdm(test_images, desc='Normalizing test images...')])

    # Compute mean and std to verify that images are normalized
    mean_train_post, std_train_post = train_images.mean(axis=[0, 2, 3]), train_images.std(axis=[0, 2, 3])
    mean_val, std_val = val_images.mean(axis=[0, 2, 3]), val_images.std(axis=[0, 2, 3])
    mean_test, std_test = test_images.mean(axis=[0, 2, 3]), test_images.std(axis=[0, 2, 3])
    
    print("")
    print(f"INFO - Training images (after normalization): \n\tMean: {mean_train_post.numpy()} | Std: {std_train_post.numpy()}")
    print(f"INFO - Validation images (after normalization): \n\tMean: {mean_val.numpy()} | Std: {std_val.numpy()}")
    print(f"INFO - Test images (after normalization): \n\tMean: {mean_test.numpy()} | Std: {std_test.numpy()}")
    print("")

    # Setup data for storage
    now = datetime.now()
    filepath = (output_path / (now.strftime("%m-%d-%Y-") + output_filename)).as_posix()
    os.makedirs(output_path, exist_ok=True)

    data = {
        'train': {
            'images': train_images,
            'image_idxs': train_idxs,
            'image_paths': train_img_paths,
            'labels': train_labels,
            'normalization': {
                'mean': mean_train,
                'std': std_train,
            },
        },
        'validation': {
            'images': val_images,
            'image_idxs': val_idxs,
            'image_paths': val_img_paths,
            'labels': val_labels,
        },
        'test': {
            'images': test_images,
            'image_idxs': test_idxs,
            'image_paths': test_img_paths,
            'labels': test_labels,
        },
        'seed': seed,
        'filepath': filepath,
        'savetime': now,
        
    }

    # Store as torch file    
    torch.save(data, filepath)

    # Check if file exists
    if os.path.isfile(filepath):
        print(f"Data saved to {filepath}")
    else:
        print(f"Failed to save data to {filepath}")

if __name__ == '__main__':
    # Fix OMP initialization error
    # os.environ['KMP_DUPLICATE_LIB_OK']='True'

    # Specify input- and output paths
    data_path = Path(r"C:\Users\carol\XAI-ResponsibleAI\data\raw\CUB_200_2011") 
    output_path = Path(r"C:\Users\carol\XAI-ResponsibleAI\data\processed\CUB_200_2011")

    # Select filename
    target_size = 224
    filename = f'processed_data_{target_size}x{target_size}.pth'

    # Create and store torch pickles
    create_pickle_files(data_path=data_path, output_filename=filename, output_path=output_path, target_size=target_size, seed=42)

