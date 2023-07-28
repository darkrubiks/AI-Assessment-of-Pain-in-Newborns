import cv2
import numpy as np
from torchvision import transforms

from dataloaders.BaseDataset import BaseDataset


class VGGNBDataset(BaseDataset):
    """
    Load the images with its respective labels. The images are normalized
    to VGGFace format.

    Parameters
    ----------
    path : the directory where the images are located

    soft : load labels as soft labels using the NFCS score
    
    cache : if True it will cache all images in RAM for faster training
    """
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224), antialias=False),
            transforms.Normalize(
                mean=[0.36703529, 0.41083294, 0.50661294], std=[1, 1, 1])
        ])
    
    def __init__(self, 
                path: str, 
                soft: bool=False,
                cache: bool=False) -> None:
        
        super().__init__(path, soft, cache)

    def load_image(self, idx):
        image = cv2.imread(self.img_names[idx]) # Load BGR image to VGGFace 
        image = image/255 # Normalize to [0-1]
        image = np.float32(image)
        image = self.transform(image) # Apply Normalization based on VGGFace

        return image