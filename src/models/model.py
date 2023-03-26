from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, resnet50, inception_v3
from torchvision.models import ResNet18_Weights, ResNet50_Weights


class ResNet50(nn.Module):
    
    def __init__(self, input_shape: Tuple[int, int]):
        super().__init__()

        self.input_shape = input_shape

        self.log_softmax = nn.LogSoftmax(dim=1)
        
    def forward(self):
        pass

def get_model(model_name: str, lr: float, device):
    if model_name not in ['ResNet18', 'ResNet50', 'Inception3']:
        raise NotImplementedError(f"No such model class exists... {(model_name)}")
    
    if model_name == 'ResNet50':
        model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
        #model = resnet18(pretrained=True)
        #model = resnet50(pretrained=True)
        # Freeze weights
        for param in model.parameters():
            param.requires_grad = False

        # Define output layers
        model.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 200),
            nn.LogSoftmax(dim=1),
        ).to(device)
         
        # Define loss criterion + optimizer --> NLLLoss used with LogSoftmax for stability reasons
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

    elif model_name == 'ResNet18':
        model = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)

        # Freeze weights
        for param in model.parameters():
            param.requires_grad = False

        # Define output layers
        model.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 200),
            nn.LogSoftmax(dim=1),
        ).to(device)
         
        # Define loss criterion + optimizer --> NLLLoss used with LogSoftmax for stability reasons
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

    elif model_name == 'Inception3':
        model = inception_v3(pretrained=True).to(device)

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
        optimizer = optim.Adam(model.parameters(), lr=lr)

    return model, criterion, optimizer

