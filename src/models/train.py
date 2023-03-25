#!/usr/bin/python
# 

import os
from pathlib2 import Path
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
import torch

import time
from tqdm import trange

from src.models.model import get_model
from src.data.dataloader import CUBDataset, get_loaders

def set_seed(seed: int):
    torch.manual_seed(seed)

#  ---------------  Training  ---------------
def train(
        datafolder_path: str, datafile_name: str,
        model_name: str,
        batch_size: int = 128, num_workers: int = 1, lr=1e-4, epochs: int = 100, 
        experiment_name: str = str(int(round(time.time()))), save_path: str = '', 
        seed: int = 42,
    ):
    
    # Set seed
    set_seed(seed)
    # Tensorboard writer for logging experiments
    writer = SummaryWriter(f"logs/{experiment_name}")

    # Get dataset splits
    loaders, normalization = get_loaders(
        data_path=datafolder_path / datafile_name,
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
    )

    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"INFO - using device: {device}")

    # Define the model, loss criterion and optimizer
    model, criterion, optimizer = get_model(model_name, lr, device=device)
    
    print("CNN Architecture:")
    print(model)

    current_best_loss = torch.inf
    with trange(epochs) as t:
        for epoch in t:
            running_loss_train, running_loss_val    = 0.0, 0.0
            running_acc_train,  running_acc_val     = 0.0, 0.0

            for batch in iter(loaders['train']):
                # Extract data                
                inputs, labels = batch['image'].to(device), batch['label'].to(device)

                if model_name == 'Inception3':
                    print("INFO - Resizing input tensor to (224, 224) for Inception3 model...")
                    # Resize the input tensor to a larger size
                    inputs = F.interpolate(inputs, size=(299, 299), mode='bilinear', align_corners=True)

                # Zero the parameter gradients
                optimizer.zero_grad()
                # Forward + backward
                outputs = model(inputs)

                if model_name == 'Inception3':
                    # Extract logits (tensor) from InceptionOutputs object
                    outputs = outputs.logits
                
                loss = criterion(outputs, labels)
                running_loss_train += loss.item()
                loss.backward()
                # Optimize
                optimizer.step()

                # Get predictions from log-softmax scores
                preds = torch.exp(outputs.detach()).topk(1)[1]
                # Store accuracy
                equals = preds.flatten() == labels
                running_acc_train += torch.mean(equals.type(torch.FloatTensor))

            # Validation
            with torch.no_grad():
                for batch in iter(loaders['validation']):
                    inputs, labels = batch['image'].to(device), batch['label'].to(device)
                    
                    if model_name == 'Inception3':
                        # Resize the input tensor to a larger size
                        inputs = F.interpolate(inputs, size=(299, 299), mode='bilinear', align_corners=True)

                    # Forward + backward
                    outputs = model(inputs)
                    preds = torch.exp(outputs).topk(1)[1]

                    if model_name == 'Inception3':
                        # Extract logits (tensor) from InceptionOutputs object
                        outputs = outputs.logits

                    # Compute loss and accuracy
                    running_loss_val += criterion(outputs, labels)
                    equals = preds.flatten() == labels
                    running_acc_val += torch.mean(equals.type(torch.FloatTensor))

            if running_loss_val / len(loaders['validation']) < current_best_loss:
                current_best_loss = running_loss_val / len(loaders['validation'])
                # Create and save checkpoint
                checkpoint = {
                    "experiment_name": experiment_name,
                    "seed": seed,
                    "model": {
                        'name': model_name,
                    },
                    "training_parameters": {
                        "save_path": save_path,
                        "lr": lr,
                        "optimizer": optimizer,
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "device": device,
                    },
                    "data": {
                        "filename": datafile_name,
                        "normalization": {
                            "mu": list(normalization['mean'].numpy()),
                            "sigma": list(normalization['std'].numpy()),
                        },
                    },
                    "best_epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                }
                os.makedirs(f"{save_path}/{experiment_name}", exist_ok=True)
                torch.save(checkpoint, f"{save_path}/{experiment_name}/best.ckpt")


            # Update progress bar
            train_loss_descr = (
                f"Train loss: {running_loss_train / len(loaders['train']):.3f}"
            )
            val_loss_descr = (
                f"Validation loss: {running_loss_val / len(loaders['validation']):.3f}"
            )
            train_acc_descr = (
                f"Train accuracy: {running_acc_train / len(loaders['train']):.3f}"
            )
            val_acc_descr = (
                f"Validation accuracy: {running_acc_val / len(loaders['validation']):.3f}"
            )
            t.set_description(
                f"EPOCH [{epoch + 1}/{epochs}] --> {train_loss_descr} | {val_loss_descr} | {train_acc_descr} | {val_acc_descr} | Progress: "
            )

            writer.add_scalar('loss/train', running_loss_train / len(loaders['train']), epoch)
            writer.add_scalar('accuracy/train', running_acc_train / len(loaders['train']), epoch)
            writer.add_scalar('loss/validation', running_loss_val / len(loaders['validation']), epoch)
            writer.add_scalar('accuracy/validation', running_acc_val / len(loaders['validation']), epoch)

if __name__ == '__main__':


    datafolder_path = Path('data/processed/CUB_200_2011')
    save_path = Path('models')

    train(
        datafolder_path=datafolder_path,
        model_name='Inception3',
        datafile_name='03-25-2023-processed_data_224x224.pth',
        batch_size=128,
        epochs=10,
        lr=1e-4,
        experiment_name='Inception3-test-10epochs',
        save_path='models',
    )