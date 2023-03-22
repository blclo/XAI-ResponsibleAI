from pathlib2 import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torchvision.io import read_image

from PIL import Image
import torchvision.transforms as T

from tqdm import tqdm

def get_meta(data_path):

    with open(data_path / 'classes.txt', 'r') as file:
        lines = file.readlines()

        info_ = [line.split() for line in lines]

    idx2label = {int(class_idx): label for class_idx, label in info_}
    label2idx = {label: int(class_idx) for class_idx, label in info_}
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

def plot_example_images(images, labels, example_idxs, idx2label, type='train'):
    fig, axs = plt.subplots(2, 5, figsize=(10, 5))
    for i in range(5):
        axs[0, i].imshow(images[example_idxs[i], :, :, :].squeeze().permute(1, 2, 0))
        axs[0, i].set_title(idx2label[labels[example_idxs[i]]])

        axs[1, i].imshow(images[example_idxs[5+i], :, :, :].squeeze().permute(1, 2, 0))
        axs[1, i].set_title(idx2label[labels[example_idxs[5+i]]])
    fig.suptitle(f'Visual inspection of {type}loader')
    plt.tight_layout()
    plt.show()
    
def create_pickle_files(data_path, output_path, target_size=256, show_examples=True):
    
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
    df = load_txt_files(data_path)
    df_train = df.query('is_training_image == 1')
    df_test = df.query('is_training_image == 0') # TODO: split test into validation also!

    # Prepare for loading training data
    train_images, train_labels, train_idxs, train_img_paths = [], [], [], []
    # Loop through training images
    for idx, row in tqdm(df_train.iterrows(), total=len(df_train)):
        # Get path
        img = Image.open((data_path / 'images' / row['image_path']).as_posix())
        # Preprocess image
        x = preprocess(img)

        if x.shape[0] == 3: # Only store color images
            train_idxs.append(idx)
            train_labels.append(row['label'])
            train_images.append(preprocess(img))
            train_img_paths.append(row['image_path'])

    #TODO: load test data and validation data


    train_images = torch.stack(train_images)

    # Do a visual inspection to see if the loading of files works
    if show_examples:
        example_idxs = np.random.choice(np.arange(train_images.__len__()), 10)
        plot_example_images(train_images, train_labels, example_idxs, idx2label)



    ### Normalization ### 
    # Compute mean and standard deviation of training set for normalization
    mean_, std_ = train_images.mean(axis=[0, 2, 3]), train_images.std(axis=[0, 2, 3])

    normalize = T.Compose(
        [   
            T.Normalize(mean=list(mean_.numpy()), std=list(std_.numpy())),
        ]
    )

    train_images = torch.stack([normalize(img) for img in tqdm(train_images)])
    
    data = {
        'train': {
            'images': train_images,
            'image_idxs': train_idxs,
            'image_paths': train_img_paths,
            'labels': train_labels,
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
    }
    
    torch.save()

    
if __name__ == '__main__':
    
    seed = 42
    torch.manual_seed(seed)

    data_path = Path(r"C:\Users\alber\Desktop\DTU\2_HCAI\ResponsibleAI\projects\xai\XAI-ResponsibleAI\data\raw\CUB_200_2011\CUB_200_2011") 
    output_path = Path(r"C:\Users\alber\Desktop\DTU\2_HCAI\ResponsibleAI\projects\xai\XAI-ResponsibleAI\data\processed\CUB_200_2011")

    # Create and store torch pickles
    create_pickle_files(data_path=data_path, output_path=output_path, target_size=256)

