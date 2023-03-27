from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, resnet50, inception_v3
from torchvision.models import ResNet18_Weights, ResNet50_Weights, Inception_V3_Weights

def get_model(model_name: str, device, lr: Optional[float] = None):
    if model_name not in ['ResNet18', 'ResNet50', 'Inception3']:
        raise NotImplementedError(f"No such model class exists... {(model_name)}")

    if model_name == 'ResNet50':
        model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
    
    elif model_name == 'ResNet18':
        model = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)

    elif model_name == 'Inception3':
        model = inception_v3(weights=Inception_V3_Weights.DEFAULT).to(device)
        # model = inception_v3(pretrained=True).to(device)

    # Freeze weights
    for param in model.parameters():
        param.requires_grad = False

    # Define output layers
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Linear(512, 200),
        nn.LogSoftmax(dim=1)
    ).to(device)

    # Define loss criterion + optimizer --> NLLLoss used with LogSoftmax for stability reasons
    criterion = nn.NLLLoss()

    if lr is not None: # For training mode
        optimizer = optim.Adam(model.parameters(), lr=lr)
        return model, criterion, optimizer
    
    else: # For test mode
        return model, criterion

