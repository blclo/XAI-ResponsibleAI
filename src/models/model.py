from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, resnet50, inception_v3
from torchvision.models import ResNet18_Weights, ResNet50_Weights, Inception_V3_Weights

def get_model(model_name: str, device, lr: Optional[float] = None, weight_decay=0.9, out_dim=200):
    """

    Args:
        model_name (str): Either the name of the model or the path to a model file.
        device (_type_): _description_
        lr (Optional[float], optional): _description_. Defaults to None.
        weight_decay (float, optional): _description_. Defaults to 0.9.
        out_dim (int, optional): _description_. Defaults to 200.

    Returns:
        _type_: _description_
    """

    if model_name == '':
        experiment = torch.load((model_name / 'best.ckpt').as_posix()) # train from checkpoint
        model_name = experiment['model']['name']
    else: # FileNotFoundError:
        experiment = None

    if model_name in {'ResNet18', 'ResNet50', 'Inception3'}:
        # Define loss criterion --> NLLLoss used with LogSoftmax for stability reasons
        criterion = nn.NLLLoss()

        if model_name == 'Inception3':
            model = inception_v3(weights=Inception_V3_Weights.DEFAULT).to(device)
        elif model_name == 'ResNet18':
            model = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
        elif model_name == 'ResNet50':
            model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
        else:
            raise NotImplementedError("Model type not available...")
        
        # Freeze weights
        # for param in model.parameters():
        #    param.requires_grad = False

        # Define output layers
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, out_dim),
            nn.LogSoftmax(dim=1)
        ).to(device)

    if experiment != None:
        model.load_state_dict(experiment['state_dict'])

    if lr is not None: # For training mode
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        return model, criterion, optimizer, num_ftrs

    else: # For test mode
        return model, criterion, None, num_ftrs


class CBM(nn.Module):

    def __init__(self):
        super().__init__()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        concept_model = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
        # Freeze weights
        for param in concept_model.parameters():
            param.requires_grad = False

        num_ftrs = concept_model.fc.in_features
        concept_model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 312),
            nn.LogSoftmax(dim=1)
        ).to(device)
        
        self.target_classifier  = nn.Sequential([
            nn.Linear(312, 200),
            nn.LogSoftmax(),
        ])

    def forward(self, x):

        concepts = self.concept_classifier(x)