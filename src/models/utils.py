import torch
from pathlib2 import Path

from src.models.model import get_model

def set_seed(seed: int):
    torch.manual_seed(seed)

def load_experiment(experiment_path: Path, device):
    
    # Load experiment
    experiment = torch.load(experiment_path.as_posix())
    
    # Print loaded experiment status
    print(f"\nINFO - LOADED EXPERIMENT: {experiment['experiment_name']}")
    print(f"INFO -  model: {experiment['model']}")
    print(f"INFO -  seed: {experiment['seed']}")
    print(f"INFO -  best epoch: {experiment['best_epoch']}")
    print(f"INFO -  data: ")
    print(f"INFO -    filename: {experiment['data']['filename']}")
    print(f"INFO -    normalization (mu, sigma): {experiment['data']['normalization']['mu'], experiment['data']['normalization']['sigma']}")
    print("")

    # Load model weights
    model, criterion = get_model(experiment['model']['name'], device=device)
    model.load_state_dict(experiment['state_dict'])

    return model, criterion