import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from dataloaders.presets import PresetTransform
from PIL import Image

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
        self.model_name = model_name.upper()
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        self.transform = PresetTransform(self.model_name).transforms

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path).convert("RGB")

        if self.model_name == 'VGGFACE':
            # For VGGFace, convert image mode as required.
            image = Image.fromarray(np.array(image)[:, :, ::-1])

        image = self.transform(image)

        blank = True if "blank" in self.img_names[idx] else False
        
        return image, blank, self.img_names[idx]