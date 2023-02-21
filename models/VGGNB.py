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
    def __init__(self, fine_tune_conv=False):
        super(VGGNB, self).__init__()

        self.fine_tune_conv = fine_tune_conv
        self.VGGFace = torch.load(os.path.join('models','VGG_face_original_model.pt'))

        for param in self.VGGFace.parameters():
            param.requires_grad  = False
        
        # Fine tune the last group of conv. layers.
        if self.fine_tune_conv:
            for param in self.VGGFace[17:31].parameters():
                param.requires_grad  = True

        self.VGGFaceFeatures = self.VGGFace.features

        self.classifier = nn.Sequential(
            
            nn.Linear(512 * 7 * 7, 512),

            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),

            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
            
        )

    def forward(self, x):
        x = self.VGGFaceFeatures(x)
        x = x.view(-1, 512 * 7 *7)
        x = self.classifier(x)

        return x
    
    def predict(self, x):
        return F.softmax(self.forward(x), dim=1)