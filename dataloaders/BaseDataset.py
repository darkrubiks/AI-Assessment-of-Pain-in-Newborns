import gc
import glob
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """
    Base Class for loading the dataset.

    Parameters
    ----------
    path : the directory where the images are located

    soft : load labels as soft labels using the NFCS score [sigmoid, step, or linear]
    
    cache : if True it will cache all images in RAM for faster training
    """
    def __init__(self, 
                path: str,
                soft: str='None',
                cache: bool=False) -> None:
        self.path = path
        self.cache = cache
        self.soft = soft
        self.images_cached = []
        self.labels_cached = []
        
        # Make sure to have this file in your main directory
        self.dataframe = pd.read_csv('iCOPE+UNIFESP_data.csv',
                                     usecols=['new_file_name','NFCS', 'class'])

        # Get only the files with *.jpg extension
        if self.soft:
            self.img_names = glob.glob(os.path.join(self.path, '*_UNIFESP_*.jpg'))
        else:
            self.img_names = glob.glob(os.path.join(self.path, '*.jpg'))
    
        # Cache images on RAM
        if self.cache:
            print(f'Caching images and labels please wait....')
            for i in range(len(self.img_names)):
                self.images_cached.append(self.load_image(i))
                self.labels_cached.append(self.load_label(i))

    def load_image(self, idx):
        raise Exception("Not Implemented")
    
    def load_label(self, idx):
        new_file_name = self.img_names[idx].split(os.sep)[-1]
        # Filter out the augmented prefix
        if 'AUG' in new_file_name:
            new_file_name = '_'.join(new_file_name.split('_')[2:])

        dataframe_result = self.dataframe[self.dataframe['new_file_name']==new_file_name]
        classe = dataframe_result['class'].values[0]

        # Get NFCS score
        NFCS = dataframe_result['NFCS'].values[0]

        # Soft-Label methods
        if self.soft == "sigmoid":
            S_x = 1 / (1 + np.exp(-NFCS + 2.5)) 
            label = torch.tensor(S_x)

        elif self.soft == "linear":
            S_x = 0.2 * NFCS 
            label = torch.tensor(S_x)

        elif self.soft == "step":
            if NFCS <= 1:
                label = torch.tensor(0.0)
            elif 2 <= NFCS < 3:
                label = torch.tensor(0.3)
            elif 3 <= NFCS < 4:
                label = torch.tensor(0.7)
            elif NFCS >= 4:
                label = torch.tensor(1.0)
        else:
            # Label encoding
            label = 1 if classe == 'pain' else 0

        return label
    
    def __del__(self):
        del self.labels_cached
        del self.images_cached
        gc.collect()

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        if self.cache:
            image = self.images_cached[idx]
            label = self.labels_cached[idx]
        else:
            image = self.load_image(idx)
            label = self.load_label(idx)

        return {'image':image, 'label':label}