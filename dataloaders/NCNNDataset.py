import os
import glob
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms   


class NCNNDataset(Dataset):
    """
    Load only the images with its respective labels. The images are resized and 
    normalized according to the NCNN original implementation.
    """
    def __init__(self, 
                img_dir: str, 
                fold: str,
                mode: str) -> None:
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
            transforms.Resize((120,120))
        ])

    def __len__(self):
        return len(os.listdir(self.path)) - 1 #-1 because of the landmarks folder

    def __getitem__(self, idx):
        image = cv2.imread(self.img_names[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image/255 # Normalize to [0-1]
        image = np.float32(image)
        image = self.transform(image)

        label = self.labels[idx]
        
        return {'image':image, 'label':label}