"""
dataset_maker.py

Author: Leonardo Antunes Ferreira
Date: 01/02/2023

dataset_maker.py contains all the Dataset Classes needed to load the newborn
images. 
"""
import os
import glob
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class VGGNBDataset(Dataset):
    """
    Load only the images with its respective labels. The images are normalized
    to VGGFace format.
    """

    def __init__(self, img_dir, fold, mode):
        self.img_dir = img_dir
        self.fold = fold
        self.mode = mode

        # Load directory for train or test
        if self.mode == 'Train':
            self.path = os.path.join(self.img_dir, self.fold, 'Train')
        else:
            self.path = os.path.join(self.img_dir, self.fold, 'Test')

        # Get only the files with *.jpg extension
        self.img_names = glob.glob(os.path.join(self.path, '*.jpg'))
        # Label encoding
        self.labels = [0 if img_name.rsplit('_', 2)[-2]=='no' else 1 for img_name in self.img_names]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.36703529, 0.41083294, 0.50661294], std=[1, 1, 1])
        ])

    def __len__(self):
        return len(os.listdir(self.path)) - 1 #-1 because of the landmarks folder

    def __getitem__(self, idx):
        image = cv2.imread(self.img_names[idx]) # Load BGR image to VGGFace 
        image = image/255 # Normalize to [0-1]
        image = np.float32(image)
        image = self.transform(image) # Apply Normalization based on VGGFace

        label = self.labels[idx]
        # Return a dictionary, later this dict can be updated to include more
        # information
        return {'image': image, 'label':label}
