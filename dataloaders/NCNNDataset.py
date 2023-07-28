import cv2
import numpy as np
from torchvision import transforms

from dataloaders.BaseDataset import BaseDataset


class NCNNDataset(BaseDataset):
    """
    Load the images with its respective labels. The images are resized and 
    normalized according to the NCNN original implementation.

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
        
        super().__init__(path, soft, cache)
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((120,120), antialias=False)
        ])


    def load_image(self, idx):
        image = cv2.imread(self.img_names[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Load RGB for NCNN
        image = image/255 # Normalize to [0-1]
        image = np.float32(image)
        image = self.transform(image)

        return image