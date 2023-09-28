import cv2
import numpy as np
from torchvision import transforms

from dataloaders.BaseDataset import BaseDataset


class ViTNBDataset(BaseDataset):
    """
    Load the images with its respective labels. The images are normalized
    to IMAGENET format.

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
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __init__(self, 
                path: str, 
                soft: bool=False,
                cache: bool=False) -> None:
        
        super().__init__(path, soft, cache)

    def load_image(self, idx):
        image = cv2.imread(self.img_names[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Load RGB for ViT
        image = image/255 # Normalize to [0-1]
        image = np.float32(image)
        image = self.transform(image) # Apply Normalization based on IMAGENET

        return image