import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import logging
from typing import Tuple, List

class TrainInfo:

    train_losses = []
    test_losses = []

def train_epoch(model: nn.Module, 
                dataloader: DataLoader, 
                criterion: nn.Module, 
                optimizer: optim.Optimizer) -> float:

    model.train()
    total_loss = 0

    for context, target in dataloader:

        output = model(context)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss

    eff_loss = total_loss / len(dataloader)
    TrainInfo.train_losses.append(eff_loss)
    return eff_loss

def test_epoch(model: nn.Module, 
               dataloader: DataLoader, 
               criterion: nn.Module) -> float:
    
    model.eval()
    total_loss = 0
    
    with torch.no_grad():

        for context, target in dataloader:

            output = model(context)
            loss = criterion(output, target)
            total_loss += loss.item()
        
        eff_loss = total_loss / len(dataloader)

    TrainInfo.test_losses.append(eff_loss)
    return eff_loss