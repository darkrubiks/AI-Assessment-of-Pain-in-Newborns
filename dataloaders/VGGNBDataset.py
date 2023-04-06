import os
import gc
import glob
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class VGGNBDataset(Dataset):
    """
    Load only the images with its respective labels. The images are normalized
    to VGGFace format.

    Args:
        img_dir: The directory where the images are located
        fold: Number of the Fold to run on
        mode: Train or Test mode
        cache: If True it will cache all images in RAM for faster training
    """
    def __init__(self, 
                img_dir: str, 
                fold: str,
                mode: str,
                cache: bool=False) -> None:
        self.img_dir = img_dir
        self.fold = fold
        self.mode = mode
        self.cache = cache
        self.images_cached = []

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
            transforms.Resize((224,224)),
            transforms.Normalize(
                mean=[0.36703529, 0.41083294, 0.50661294], std=[1, 1, 1])
        ])

        # Cache images on RAM
        if self.cache:
            print(f'Caching {self.mode} images please wait....')
            for i in range(len(self.img_names)):
                self.images_cached.append(self.load_image(i))

    def load_image(self, idx):
        image = cv2.imread(self.img_names[idx]) # Load BGR image to VGGFace 
        image = image/255 # Normalize to [0-1]
        image = np.float32(image)
        image = self.transform(image) # Apply Normalization based on VGGFace

        return image
    
    def __del__(self):
        del self.labels
        del self.images_cached
        gc.collect()

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        label = self.labels[idx]
        
        if self.cache:
            image = self.images_cached[idx]
        else:
            image = self.load_image(idx)

        return {'image':image, 'label':label}