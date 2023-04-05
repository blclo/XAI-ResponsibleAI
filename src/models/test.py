import time

import torch
import numpy as np
from tqdm import tqdm

from src.data.dataloader import get_loaders
from src.models.utils import set_seed, load_experiment

def test_model(
        loaders: dict,
        model, criterion,
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        seed: int = 42,
    ):
    
    # Set seed
    set_seed(seed)
    # Device
    print(f"INFO - using device: {device}")

    # Initialize storage parameters
    predictions, probabilities, targets = [], [], []
    equals = []
    batch_losses = []

    model.eval()
    # Testing loop
    with torch.no_grad():
        for batch in tqdm(iter(loaders['test']), desc='Predicting on test set...'):
            inputs, labels, concepts = batch
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward + backward
            outputs = model(inputs)
            probs = torch.exp(outputs)
            preds = probs.topk(1)[1]

            # Store predictions
            probabilities.extend(probs)
            predictions.extend(preds)
            targets.extend(labels)

            # Get predictions
            batch_losses.append(criterion(outputs, labels))
            equals.extend(preds.flatten() == labels)

    print(f"{'-'*80}\nPERFORMANCE ON TEST SET\n{'-'*80}")
    equals = torch.stack(equals).cpu().numpy()
    acc, sem = equals.mean(), equals.std() / np.sqrt(equals.__len__())
    print(f"Avg. accuracy (with SEM) =  {acc:.5f} +- {sem:.5f}")
    print(f"{'-'*80}")

    return torch.stack(predictions), torch.stack(probabilities), torch.stack(targets)



def test_bottleneck_model(
        loaders: dict,
        model, criterion,
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        seed: int = 42,
    ):
    
    # Set seed
    set_seed(seed)
    # Device
    print(f"INFO - using device: {device}")

    # Initialize storage parameters
    predictions, probabilities, targets = [], [], []
    equals = []
    batch_losses = []

    # Testing loop
    with torch.no_grad():
        for batch in tqdm(iter(loaders['test']), desc='Predicting on test set...'):
            inputs, labels, concepts = batch
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward + backward
            outputs = model(inputs)
            probs = torch.exp(outputs)
            preds = probs.topk(1)[1]

            # Store predictions
            probabilities.extend(probs)
            predictions.extend(preds)
            targets.extend(labels)

            # Get predictions
            batch_losses.append(criterion(outputs, labels))
            equals.extend(preds.flatten() == labels)

    print(f"{'-'*80}\nPERFORMANCE ON TEST SET\n{'-'*80}")
    equals = torch.stack(equals).cpu().numpy()
    acc, sem = equals.mean(), equals.std() / np.sqrt(equals.__len__())
    print(f"Avg. accuracy (with SEM) =  {acc:.5f} +- {sem:.5f}")
    print(f"{'-'*80}")

    return torch.stack(predictions), torch.stack(probabilities), torch.stack(targets)


