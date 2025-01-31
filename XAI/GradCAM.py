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
import numpy as np
import torch
import torch.nn.functional as F

class GradCAM:
    """
    GradCAM method optimized for batch inputs and faster resizing.

    This implementation produces heatmaps that indicate which regions of the
    input image contributed most strongly to the output. The code is optimized
    for batch inputs and uses PyTorch’s F.interpolate for GPU-accelerated resizing.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be explained.
    target_layer : torch.nn.Module
        The layer whose activations/gradients are used for CAM computation.
    device : str, optional
        Device to run the computations on ('cpu' or 'cuda'), by default 'cpu'.
    reshape_transform_ViT : bool, optional
        Whether to apply a reshape transform (e.g. for Vision Transformers),
        by default False.
    """
    def __init__(self, 
                 model: torch.nn.Module,
                 target_layer: torch.nn.Module,
                 device: str='cpu',
                 reshape_transform_ViT: bool=False) -> None:
        self.device = device
        self.model = model.eval().to(self.device)
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.reshape_transform_ViT = reshape_transform_ViT

        # Register two forward hooks:
        # - One to capture activations (feature maps)
        # - One to register a hook on the output to capture gradients during backprop.
        self.f_hook = self.target_layer.register_forward_hook(self.__get_featuremaps)
        self.b_hook = self.target_layer.register_forward_hook(self.__get_gradients)
    
    def __get_featuremaps(self, module, input, output):
        """
        Stores the module's output (activations) for later use.
        If a reshape transform is required (e.g. for ViT), apply it.
        """
        if self.reshape_transform_ViT:
            self.activations = self.__reshape_transform(output)
        else:
            self.activations = output.detach()

    def __get_gradients(self, module, input, output):
        """
        Registers a hook on the module's output to capture gradients during backprop.
        Since backward hooks on non-leaf tensors are not supported reliably,
        we attach a hook on the output tensor.
        """
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            return

        def __store_grad(grad):
            if self.reshape_transform_ViT:
                self.gradients = self.__reshape_transform(grad)
            else:
                self.gradients = grad.detach()
        output.register_hook(__store_grad)

    def __reshape_transform(self, tensor, height=14, width=14):
        """
        Reshape transformer outputs (e.g., from ViT models) to 4D
        tensors with shape (B, C, height, width). This typically involves
        removing the class token and rearranging the remaining tokens.
        """
        # Assume tensor shape: (B, num_tokens, features)
        # Skip the first token (class token), then reshape.
        result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
        # Permute to (B, features, height, width) to match CNN activations.
        result = result.permute(0, 3, 1, 2)
        return result

    def attribution_mask(self, image: torch.Tensor) -> np.ndarray:
        """
        Generates GradCAM heatmaps.

        Parameters
        ----------
        image : torch.Tensor
            Input image tensor of shape (B, C, H, W) or (C, H, W). If the latter,
            a batch dimension is added.

        Returns
        -------
        np.ndarray
            A numpy array of shape (B, H, W) containing the normalized heatmaps.
        """
        # Ensure image has a batch dimension.
        if image.dim() != 4:
            image = image.unsqueeze(0)

        image = image.to(self.device).requires_grad_()
        
        # Forward pass.
        logit = self.model(image)
        
        # Zero out previous gradients.
        self.model.zero_grad()

        # Backward pass.
        # Here we backpropagate ones for each output to compute gradients.
        # (Optionally, you could pass target-specific gradients here.)
        logit.backward(torch.ones_like(logit), retain_graph=False)
        
        # Compute channel-wise weights via global average pooling over spatial dims.
        # weights shape: (B, channels)
        weights = torch.mean(self.gradients, dim=(2, 3))
        
        # Compute the weighted combination of feature maps.
        # activations shape: (B, channels, H_a, W_a)
        # After weighted sum -> cam shape: (B, H_a, W_a)
        cam = torch.sum(self.activations * weights[:, :, None, None], dim=1)
        
        # Apply ReLU (only positive influences) and normalize each CAM per sample.
        cam = F.relu(cam)
        B, H_a, W_a = cam.shape
        # Compute the maximum value for each sample.
        cam_flat = cam.view(B, -1)
        max_vals, _ = torch.max(cam_flat, dim=1, keepdim=True)
        cam = cam / (max_vals.view(B, 1, 1) + 1e-7)
        
        # Resize the CAM to match the input image size using bilinear interpolation.
        _, _, H, W = image.shape
        cam = cam.unsqueeze(1)  # add channel dimension: (B, 1, H_a, W_a)
        cam_resized = F.interpolate(cam, size=(H, W), mode='bilinear', align_corners=False)
        cam_resized = cam_resized.squeeze(1)  # final shape: (B, H, W)
        
        return cam_resized.detach().cpu().numpy()
    
    def __del__(self):
        """Remove hooks when the object is destroyed."""
        self.f_hook.remove()
        self.b_hook.remove()