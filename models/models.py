"""
models.py

Author: Leonardo Antunes Ferreira
Date:13/02/2022

This file contains all the models used during the project.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGNB(nn.Module):
    def __init__(self):
        super(VGGNB, self).__init__()

        self.VGGFace = torch.load('models\\VGG_face_original_model.pt')

        for param in self.VGGFace.parameters():
            param.requires_grad  = False

        self.VGGFaceFeatures = self.VGGFace.features

        self.Dropout = nn.Dropout(0.5)

        self.FC1 = nn.Linear(512 * 7 * 7, 512)
        self.FC2 = nn.Linear(512, 512)
        self.output = nn.Linear(512, 2)

    def forward(self, x):
        x = self.VGGFaceFeatures(x)
        x = x.view(-1, 512 * 7 *7)
        x = F.relu(self.FC1(x))
        x = self.Dropout(x)
        x = F.relu(self.FC2(x))
        x = self.Dropout(x)
        x = self.output(x)

        return x
    
    def predict(self, x):
        return F.softmax(self.forward(x))