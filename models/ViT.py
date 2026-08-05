"""
ViTNB.py

Author: Leonardo Antunes Ferreira
Date:22/09/2023

This model implements the ViT_b_32 architecture pre-trained on IMAGENET.
Only the classification head is trainable.
"""
from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ViT_B_32_Weights, vit_b_32


class ViT(nn.Module):
    def __init__(self, weights: Optional[ViT_B_32_Weights | str] = 'IMAGENET1K_V1') -> None:
        super(ViT, self).__init__()

        # ``weights=None`` is useful when a complete local checkpoint is loaded
        # immediately afterwards (for example, in the real-time benchmark).  It
        # avoids an unnecessary network/cache dependency while preserving the
        # historical pretrained default used by the training scripts.
        self.ViT = vit_b_32(weights=weights)

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
