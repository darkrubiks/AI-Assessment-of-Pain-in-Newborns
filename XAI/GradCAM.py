"""
GradCAM.py

Author: Leonardo Antunes Ferreira
Date: 04/03/2023

This file contains the GradCAM algorithm, an Explainable AI technique introduced
in the paper Grad-CAM: Visual Explanations from Deep Networks via Gradient-based
Localization. GradCAM aims to produce a coarse localization map highlighting the
important regions in the image for predicting the concept.

doi: https://doi.org/10.48550/arXiv.1610.02391

This code is heavily inspired by https://github.com/jacobgil/pytorch-grad-cam
"""
import cv2
import torch
import torch.nn.functional as F
import numpy as np


class GradCAM:
    def __init__(self, 
                 model: torch.nn.Module,
                 target_layer: torch.nn.Module) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.eval().to(self.device)
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.f_hook = self.target_layer.register_forward_hook(self.__get_featuremaps)
        # Because of https://github.com/pytorch/pytorch/issues/61519,
        # we don't use backward hook to record gradients
        self.b_hook = self.target_layer.register_forward_hook(self.__get_gradients)
    
    def __get_featuremaps(self, module, input, output):
        self.activations = output.detach()

    def __get_gradients(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad
            return
        
        def __store_grad(grad):
            self.gradients = grad.detach()

        output.register_hook(__store_grad)

    def attribution_mask(self, 
                         image: torch.Tensor, 
                         target_class: int):
        """
        Generates the GradCAM attribution mask. The image needs to be in batch
        format like (1, 3, 224, 224).
        """
        image = image.to(self.device).requires_grad_()
        logit = self.model(image)[:, target_class]
    
        self.model.zero_grad()

        logit.backward(retain_graph=True)

        # Neuron importance weights, apply ReLU to the weights because we are 
        # only interested in the positive influence on the class of interest
        weights = F.relu(torch.mean(self.gradients, axis=(2,3)))

        # Weighted combination of forward activation maps
        cam = torch.sum(self.activations * weights[:, :, None, None], axis=1)
        cam = torch.clamp(cam, min=0)
        cam = cam/torch.max(cam)
        cam = cam.cpu().numpy().squeeze()

        return self.resize_mask(image, cam)

    def resize_mask(self,
                    image: torch.Tensor,
                    cam: np.ndarray):
        """
        Resize the CAM mask to the original image size.
        """
        width, height = image.size(-1), image.size(-2)
        resized_cam = cv2.resize(cam, (width, height))

        return resized_cam
        
    def __del__(self):
        """
        Remove the hooks.
        """
        self.f_hook.remove()
        self.b_hook.remove()