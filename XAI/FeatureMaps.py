"""
FeatureMaps.py

Author: Leonardo Antunes Ferreira
Date: 17/06/2023

This file contains The FeatureMaps class which return the feature maps of any
convolutional layer of your model. Feature maps are import to understand the
feedforward process of your model.
"""
import torch
import numpy as np


class FeatureMaps:
    """
    Feature Maps are the result of an image passing through a convolutional layer
    filters.

    Parameters
    ----------
    model : the model to be used

    target_layer : the target layer to be used

    Returns
    -------
    feature_maps : a numpy array of feature maps
    """
    def __init__(self, 
                 model: torch.nn.Module,
                 target_layer: torch.nn.Module) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.eval().to(self.device)
        self.target_layer = target_layer
        self.feature_maps = None

        self.f_hook = self.target_layer.register_forward_hook(self.__forward_pass)
        
    def __forward_pass(self, module, input, output):
        self.feature_maps = output.detach()

    def get_featuremaps(self, 
                        image: torch.Tensor) -> np.ndarray:
        """
        Generates the feature maps.
        """
        if len(image.shape) != 4:
            # Makes sure image is in batch shape.
            image = image.unsqueeze(0)

        image = image.to(self.device)
        _ = self.model(image)
        self.feature_maps = torch.clamp(self.feature_maps, min=0)
        self.feature_maps = self.feature_maps.detach().numpy().squeeze()

        return self.feature_maps
        
    def __del__(self):
        """
        Remove the hook.
        """
        self.f_hook.remove()