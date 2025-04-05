"""
ViTNB.py

Author: Leonardo Antunes Ferreira
Date:22/09/2023

This model implements the ViT_b_16 architecture pre-trained on IMAGENET.
Only the classification head is trainable.
"""
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16


class ViT(nn.Module):
    def __init__(self) -> None:
        super(ViT, self).__init__()

        self.ViT = vit_b_16(weights='IMAGENET1K_V1')

        for param in self.ViT.parameters():
            param.requires_grad  = False

        # Get the number of input features for the classification head
        in_features = self.ViT.heads.head.in_features  
        self.ViT.heads.head = nn.Linear(in_features, 1)

    def forward(self, x):
        x = self.ViT(x)
        x = x.view(-1)
        return x
    
    def predict(self, x):
        return F.sigmoid(self.forward(x))