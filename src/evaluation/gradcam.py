import torch
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def get_vizualisation_batch(loaders, preds, selected_idxs: list):

    # Construct "batch" for visualization
    batch = {'index': [], 'image': [], 'label': [], 'prediction': [], 'concepts': []}
    for i in selected_idxs:
        img, lab, concept = loaders['test'].dataset.__getitem__(i)
        batch['index'].append(i)
        batch['image'].append(img)
        batch['label'].append(lab)
        batch['prediction'].append(preds[i])
        batch['concepts'].extend(concept)

    # Stack data
    batch['image'] = torch.stack(batch['image'])
    batch['label'] = torch.tensor(batch['label']).reshape(-1, 1)
    return batch

def compute_saliency_maps(batch: dict, cam, CAM_target: str = 'prediction', device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

    # Extract information
    input_tensor    = batch['image']
    labels          = batch['label']
    predictions     = batch['prediction']

    # Add gradient for backtracking
    input_tensor = input_tensor.to(device)
    input_tensor.requires_grad_()

    # GradCAM target
    if CAM_target == 'label':
        targets = [ClassifierOutputTarget(label) for label in labels]
    elif CAM_target == 'prediction':
        targets = [ClassifierOutputTarget(pred) for pred in predictions]
    else:
        targets = None

    # Compute GradCAM saliency map 
    grayscale_maps = cam(input_tensor=input_tensor, targets=targets)
    return {idx: grayscale_maps[i, :] for i, idx in enumerate(batch['index'])}


def visualize_saliency_maps(saliency_maps, batch, normalization, meta, ncols=10, figsize=(8, 4)):
    
    # Inverse normalization - reverts from normalized to original image space
    invNormalization = transforms.Compose([
        transforms.Normalize(mean=[ 0., 0., 0. ], std=1 / normalization['std']), 
        transforms.Normalize(mean=-normalization['mean'], std = [1., 1., 1.]),
    ])
    
    # Extract information
    input_tensor    = batch['image']
    labels          = batch['label']
    preds           = batch['prediction'] 

    N = saliency_maps.items().__len__() 
    nrows = int(np.ceil(N / ncols))

    # Plot saliency maps on images
    fig, axs = plt.subplots(nrows, ncols, sharex=True, figsize=figsize, squeeze=False)
    print(len(saliency_maps.items()))
    for i, (idx, s_map) in enumerate(saliency_maps.items()):

        # Revert the normalization
        reverted_img = invNormalization(input_tensor[i]).permute(1, 2, 0).numpy()
        # Show image
        axs[i // ncols, i % ncols].imshow(reverted_img)

        # Layover saliency map
        axs[i // ncols, i % ncols].imshow(s_map, alpha=0.6)
        axs[i // ncols, i % ncols].axis('off')
        axs[i // ncols, i % ncols].set_title(
            "$\mathbf{Index}$: " + f"{idx}" + "\n$\mathbf{y}$: " + 
            f"{meta['classes']['idx2label'][labels[i].item()]}" + 
            "\n$\mathbf{\hat y}$: " + f"{meta['classes']['idx2label'][preds[i].item()]}", 
            loc='left')

    return fig

def visualize_saliency_maps_carol(saliency_maps, batch, normalization, meta, ncols=5, figsize=(30, 5)):
    
    # Inverse normalization - reverts from normalized to original image space
    invNormalization = transforms.Compose([
        transforms.Normalize(mean=[ 0., 0., 0. ], std=1 / normalization['std']), 
        transforms.Normalize(mean=-normalization['mean'], std = [1., 1., 1.]),
    ])
    
    # Extract information
    input_tensor    = batch['image']
    labels          = batch['label']
    preds           = batch['prediction'] 

    N = saliency_maps.items().__len__() # 5

    # Plot saliency maps on images
    fig, axs = plt.subplots(1, ncols, sharex=True, figsize=figsize, squeeze=False)
    for i, (idx, s_map) in enumerate(saliency_maps.items()):

        # Revert the normalization
        reverted_img = invNormalization(input_tensor[i]).permute(1, 2, 0).numpy()
        # Show image
        axs[i].imshow(reverted_img)

        # Layover saliency map
        axs[i].imshow(s_map, alpha=0.6)
        axs[i].axis('off')
        axs[i].set_title(
            "$\mathbf{Index}$: " + f"{idx}" + "\n$\mathbf{y}$: " + 
            f"{meta['classes']['idx2label'][labels[i].item()]}" + 
            "\n$\mathbf{\hat y}$: " + f"{meta['classes']['idx2label'][preds[i].item()]}", 
            loc='left')

    return fig
