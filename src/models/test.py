import time

import torch
import numpy as np
from tqdm import tqdm

from src.data.dataloader import get_loaders
from src.models.utils import set_seed, load_experiment

def test(
        datafolder_path: str, datafile_name: str,
        experiment_path: str,
        batch_size: int = 128, num_workers: int = 1, 
        seed: int = 42,
    ):
    
    # Set seed
    set_seed(seed)
    
    # Get dataset splits
    loaders, _ = get_loaders(
        data_path=datafolder_path / datafile_name,
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
    )

    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"INFO - using device: {device}")

    # Load experiment stored from training
    model_name, model, criterion = load_experiment(experiment_path, device=device)
    model.eval()

    print(model)
    
    predictions = []
    equals = []
    batch_losses = []

    # Testing loop
    with torch.no_grad():
        for batch in tqdm(iter(loaders['test']), desc='Predicting on test set...'):
            inputs, labels = batch['image'].to(device), batch['label'].to(device)
            
            # Forward + backward
            outputs = model(inputs)
            preds = torch.exp(outputs).topk(1)[1]
            predictions.extend(preds)

            # Get predictions
            batch_losses.append(criterion(outputs, labels))
            equals.extend(preds.flatten() == labels)

    print(f"\n{'-'*80}\nPERFORMANCE ON TEST SET\n{'-'*80}")
    preds = torch.stack(equals).cpu().numpy()
    acc, sem = preds.mean(), preds.std() / np.sqrt(preds.__len__())
    print(f"Avg. accuracy (with SEM) =  {acc:.5f} +- {sem:.5f}")
    print(f"{'-'*80}\n")

if __name__ == '__main__':

    from pathlib2 import Path

    # BASE_PATH = Path('projects/xai/XAI-ResponsibleAI')
    BASE_PATH = Path()

    datafolder_path = BASE_PATH / 'data/processed/CUB_200_2011'
    
    # experiment_path = BASE_PATH / 'models/ResNet18-test-50epochs/best.ckpt'
    experiment_path = BASE_PATH / 'models/Inception-test-new-dummy/best.ckpt'

    # Run test
    test(
        datafolder_path=datafolder_path,
        datafile_name='03-24-2023-processed_data_224x224.pth',
        batch_size=64,
        experiment_path=experiment_path,
    )
