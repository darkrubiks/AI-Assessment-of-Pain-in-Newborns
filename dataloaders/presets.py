"""
Module providing utilities for soft labeling and preset image transformations.

Classes:
    SoftLabel: Computes soft labels from NFCS scores using specified methods.
    PresetTransform: Provides preset torchvision transforms based on model architecture.
"""

import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


class SoftLabel:
    """
    Utility class for computing soft labels based on a given NFCS score.

    Parameters
    ----------
    soft : str
        The soft labeling method to use. Supported options are "SIGMOID", "LINEAR", and "STEP".
    """

    def __init__(self, soft: str):
        self.soft = soft.upper()

    def get_soft_label(self, nfcs: float):
        """
        Compute the soft label for a given NFCS score.

        Parameters
        ----------
        nfcs : float
            The NFCS score for which to compute the soft label.

        Returns
        -------
        torch.Tensor
            The computed soft label as a tensor.
        """
        if self.soft == "SIGMOID":
            label = label = torch.sigmoid(torch.tensor(nfcs - 2.5))
        elif self.soft == "LINEAR":
            label = torch.tensor(0.2 * nfcs)
        elif self.soft == "STEP":
            if nfcs <= 1:
                label = torch.tensor(0.0)
            elif 1 < nfcs < 3:
                label = torch.tensor(0.3)
            elif 3 <= nfcs < 4:
                label = torch.tensor(0.7)
            else:  # nfcs >= 4
                label = torch.tensor(1.0)
        else:
            raise ValueError(f"Unsupported soft label method: {self.soft}")
        
        return label.type(torch.float32)


class PresetTransform:
    """
    Provides preset torchvision transforms based on the specified model architecture.

    Parameters
    ----------
    model_name : str
        The name of the model architecture. Supported values are 'NCNN', 'VGGFace', and 'ViT'.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name.upper()
        interpolation = InterpolationMode.BILINEAR

        if self.model_name == 'NCNN':
            self.transforms = transforms.Compose([
                transforms.Resize((120,120), interpolation=interpolation, antialias=True),
                transforms.ConvertImageDtype(torch.float),
            ])
        elif self.model_name == 'VGGFACE':
            self.transforms = transforms.Compose([
                transforms.Resize((224,224), interpolation=interpolation, antialias=True),
                transforms.ConvertImageDtype(torch.float),
                # VGGFace normalization
                transforms.Normalize(mean=[0.367, 0.410, 0.506],
                                     std=[1, 1, 1])
            ])
        elif self.model_name == 'VIT':
            self.transforms = transforms.Compose([
                transforms.Resize((224,224), interpolation=interpolation, antialias=True),
                transforms.ConvertImageDtype(torch.float),
                # Imagenet normalization
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")
