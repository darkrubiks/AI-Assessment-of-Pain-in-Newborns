"""
VGGNB.py

Author: Leonardo Antunes Ferreira
Date:13/02/2023

This file contains the VGGNB model implemented by:

Leonardo A. Ferreira, University Center of FEI,São Bernardo do Campo SP,Brazil  
Lucas P. Carlini, University Center of FEI,São Bernardo do Campo SP,Brazil  
Gabriel A. S. Coutrin, University Center of FEI,São Bernardo do Campo SP,Brazil  
Victor V. Varoto, University Center of FEI,São Bernardo do Campo SP,Brazil 

on "A Convolutional Neural Network-based Mobile Application to Bedside Neonatal 
Pain Assessment" in 2021 34th SIBGRAPI.

doi: https://doi.org/10.1109/SIBGRAPI54419.2021.00060
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGNB(nn.Module):
    def __init__(self) -> None:
        super(VGGNB, self).__init__()

        self.VGGFace = torch.load(os.path.join('models','VGG_face_original_model.pt'))

        for param in self.VGGFace.parameters():
            param.requires_grad  = False

        # Change classifier head to the proposed architecture.
        self.VGGFace.classifier = nn.Sequential(
            
            nn.Linear(in_features=512 * 7 * 7,
                      out_features=512),

            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=512, 
                      out_features=512),

            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=512,
                      out_features=1)
        )

    def forward(self, x):
        x = self.VGGFace(x)
        return x
    
    def predict(self, x):
        return F.sigmoid(self.forward(x))