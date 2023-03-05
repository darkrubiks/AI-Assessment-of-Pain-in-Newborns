"""
NCNN.py

Author: Leonardo Antunes Ferreira
Date: 05/03/2023

This file contains the NCNN model implemented by:

G. Zamzmi, R. Paul, D. Goldgof, R. Kasturi and Y. Sun on 

"Pain Assessment From Facial Expression: Neonatal Convolutional Neural Network (N-CNN)"

doi: https://doi.org/10.1109/IJCNN.2019.8851879
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as T


class NCNN(nn.Module):
    def __init__(self) -> None:
        super(NCNN, self).__init__()

        # Branch 1
        self.maxpool_1_1 = nn.MaxPool2d(kernel_size=10, stride=10, padding=0)
        # Branch 2
        self.conv_2_1 = nn.Conv2d(3, 64, 5, 1, padding=2)
        self.maxpool_2_1 = nn.MaxPool2d(kernel_size=3, stride=3, padding=1)
        self.conv_2_2 = nn.Conv2d(64, 64, 5, 1, padding=2)
        self.maxpool_2_2 = nn.MaxPool2d(kernel_size=3, stride=3, padding=1)
        self.dropout_2 = nn.Dropout(0.1)
        # Branch 3
        self.conv_3_1 = nn.Conv2d(3, 64, 5, 1, padding=2)
        self.maxpool_3_1 = nn.MaxPool2d(10 ,10, padding=0)
        self.dropout_3 = nn.Dropout(0.1)
        # Merge Branch
        self.conv_4 = nn.Conv2d(64 + 64 + 3, 64, 2, 1, padding=1, dilation=2)
        self.maxpool_4 = nn.MaxPool2d(2, 2, padding=0)
        self.fc_4 = nn.Linear(6 * 6 * 64, 8)
        self.dropout_4 = nn.Dropout(0.1)
        self.output = nn.Linear(8, 2)

    def branch_1(self, x):
        x = self.maxpool_1_1(x)

        return x
    
    def branch_2(self, x):
        x = F.leaky_relu(self.conv_2_1(x), 0.01)
        x = self.maxpool_2_1(x)
        x = F.leaky_relu(self.conv_2_2(x), 0.01)
        x = self.maxpool_2_2(x)
        x = T.resize(x, 12)
        x = self.dropout_2(x)

        return x

    def branch_3(self, x):
        x = F.leaky_relu(self.conv_3_1(x), 0.01)
        x = self.maxpool_3_1(x)
        x = self.dropout_3(x)

        return x
    
    def merge_branch(self, x_1, x_2, x_3):
        x = torch.cat((x_1, x_2, x_3), dim=1)
        x = F.relu(self.conv_4(x))
        x = self.maxpool_4(x)
        x = x.view(-1, 6 * 6 * 64)
        x = F.relu(self.fc_4(x))
        x = self.dropout_4(x)
        x = self.output(x)

        return x

    def forward(self, x):
        x_1 = self.branch_1(x)
        x_2 = self.branch_2(x)
        x_3 = self.branch_3(x)

        x = self.merge_branch(x_1, x_2, x_3)
        
        return x
    
    def predict(self, x):
        return F.softmax(self.forward(x), dim=1)