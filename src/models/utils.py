import torch
from pathlib2 import Path

def set_seed(seed: int):
    torch.manual_seed(seed)

def load_experiment(experiment_path: Path):

    print("Here")

    model = None
    criterion = torch.nn.NLLLoss()
    
    return model, criterion