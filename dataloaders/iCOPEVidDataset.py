import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

class iCOPEVidDataset(Dataset):
    def __init__(self, 
                 img_dir: str, 
                 model_name: str) -> None:
        """
        Class for loading the iCOPEVid dataset.

        Parameters
        ----------
        img_dir : Directory with all the images.

        model_name : Model for which we are preparing the data (e.g., 'NCNN', 'VGGNB').
        """
        self.img_dir = img_dir
        self.model_name = model_name
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        self.transform = self.get_transform(model_name)

    def get_transform(self, model_name):
        """Define different transforms based on the model selected."""
        if 'NCNN' in model_name:
            # Example transformations for NCNN model
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((120,120), antialias=False)
        ])

        elif 'VGGNB' in model_name:
            # Example transformations for VGGNB model
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224,224), antialias=False),
                transforms.Normalize(
                    mean=[0.36703529, 0.41083294, 0.50661294], std=[1, 1, 1])
        ])

        elif 'ViTNB' in model_name:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224,224), antialias=False),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = cv2.imread(img_path)

        if self.model_name == "NCNN" or self.model_name == "ViTNB":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = image / 255 # Normalize to [0-1]
        image = np.float32(image)

        if self.transform:
            image = self.transform(image)

        blank = True if "blank" in self.img_names[idx] else False
        
        return image, blank, self.img_names[idx]