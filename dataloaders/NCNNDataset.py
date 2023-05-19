import os
import gc
import glob
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms   


class NCNNDataset(Dataset):
    """
    Load only the images with its respective labels. The images are resized and 
    normalized according to the NCNN original implementation.

    Parameters
    ----------
    img_dir : the directory where the images are located

    fold : number of the Fold to run on

    mode : train or Test mode

    soft : load labels as soft labels using the NFCS score
    
    cache : if True it will cache all images in RAM for faster training
    """
    def __init__(self, 
                img_dir: str, 
                fold: str,
                mode: str,
                soft: bool=False,
                cache: bool=False) -> None:
        self.img_dir = img_dir
        self.fold = fold
        self.mode = mode
        self.soft = soft
        self.cache = cache
        self.images_cached = []
        self.labels_cached = []
        # Make sure to have this file on your main directory
        self.dataframe = pd.read_csv('iCOPE+UNIFESP_data.csv',
                                     usecols=['new_file_name','NFCS', 'class'])
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((120,120))
        ])

        # Load directory for train or test
        if self.mode == 'Train':
            self.path = os.path.join(self.img_dir, self.fold, 'Train')
        else:
            self.path = os.path.join(self.img_dir, self.fold, 'Test')

        # Get only the files with *.jpg extension
        self.img_names = glob.glob(os.path.join(self.path, '*.jpg'))

        # Cache images on RAM
        if self.cache:
            print(f'Caching {self.mode} images and labels please wait....')
            for i in range(len(self.img_names)):
                self.images_cached.append(self.load_image(i))
                self.labels_cached.append(self.load_label(i))

    def load_image(self, idx):
        image = cv2.imread(self.img_names[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Load RGB fro NCNN
        image = image/255 # Normalize to [0-1]
        image = np.float32(image)
        image = self.transform(image)

        return image
    
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