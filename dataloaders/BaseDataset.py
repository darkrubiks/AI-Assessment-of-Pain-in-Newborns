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

    soft : load labels as soft labels using the NFCS score
    
    cache : if True it will cache all images in RAM for faster training
    """
    def __init__(self, 
                path: str,
                soft: bool=False,
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
            print(f'Caching {self.mode} images and labels please wait....')
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

        if self.soft:
            # Transform the NFCS into a soft label using the sigmoid function
            NFCS = dataframe_result['NFCS'].values[0]
            S_x = 1 / (1 + np.exp(-NFCS + 2.5))
            label = torch.Tensor([1 - S_x, S_x])
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