"""
ViTNB.py

Author: Leonardo Antunes Ferreira
Date:22/09/2023


"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16


class ViTNB(nn.Module):
    def __init__(self) -> None:
        super(ViTNB, self).__init__()

        self.ViT = vit_b_16(weights='IMAGENET1K_V1')

        for param in self.ViT.parameters():
            param.requires_grad  = False

        in_features = self.ViT.heads.head.in_features  # Get the number of input features for the classification head
        self.ViT.heads.head = nn.Linear(in_features, 1)

    def forward(self, x):
        x = self.ViT(x)
        x = x.view(-1)
        return x
    
    def predict(self, x):
        return F.sigmoid(self.forward(x))