"""
IntegratedGradients.py

Author: Leonardo Antunes Ferreira
Date:20/02/2023

This file contains the Integrated Gradients algorithm, an Explainable AI techni-
que introduced in the paper Axiomatic Attribution for Deep Networks. IG aims to 
explain the relationship between a model's predictions in terms of its features.
It has many use cases including understanding feature importances, identifying 
data skew, and debugging model performance.

doi: https://doi.org/10.48550/arXiv.1703.01365
"""
import numpy as np
import torch

class IntegratedGradients:
    """
    Integrated Gradients method optimized for batched inputs.

    This implementation approximates the integral along the straight-line path 
    from a baseline (zero image) to the input image using the Riemann Trapezoidal rule.
    The number of interpolation steps is fixed (default: 50), as experiments suggest
    that increasing or decreasing the step count does not significantly affect results.
    
    Parameters
    ----------
    model : torch.nn.Module
        The model whose predictions we wish to explain.
    device : str, optional
        Device to run the computations on ('cpu' or 'cuda'), by default 'cpu'.
    n_steps : int, optional
        Fixed number of interpolation steps (default: 50).
    
    See Also
    --------
    Sundararajan, Mukund, Ankur Taly, and Qiqi Yan. "Axiomatic attribution for 
    deep networks." International conference on machine learning. PMLR, 2017.
        doi: https://doi.org/10.48550/arXiv.1703.01365
    """
    def __init__(self, model: torch.nn.Module, device: str = 'cpu', n_steps: int = 1) -> None:
        self.device = device
        self.model = model.eval().to(self.device)
        self.n_steps = n_steps

    def __interpolate_image(self, baseline: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """
        Generates a linear interpolation between the baseline and the original image,
        handling batched inputs. The number of steps is fixed (self.n_steps).
        
        Parameters
        ----------
        baseline : torch.Tensor
            Tensor of shape (B, C, H, W) representing the baseline images.
        image : torch.Tensor
            Tensor of shape (B, C, H, W) representing the input images.

        Returns
        -------
        torch.Tensor
            Interpolated images of shape (n_steps+1, B, C, H, W).
        """
        n_steps = self.n_steps
        # Create alphas with shape (n_steps+1, 1, 1, 1, 1) for proper broadcasting.
        alphas = torch.linspace(0.0, 1.0, steps=n_steps+1, device=self.device).view(n_steps+1, 1, 1, 1, 1)
        # Add an interpolation dimension to baseline and image: (1, B, C, H, W)
        baseline = baseline.unsqueeze(0)
        image = image.unsqueeze(0)
        # Interpolate along the new dimension.
        interpolated = baseline + alphas * (image - baseline)
        return interpolated

    def __compute_gradients(self, images: torch.Tensor, chunk_size: int = 10) -> torch.Tensor:
        """
        Calculates gradients for each interpolated image in smaller chunks to reduce memory usage.
        
        Parameters
        ----------
        images : torch.Tensor
            Tensor of shape (n_steps+1, B, C, H, W) representing the interpolated images.
        chunk_size : int, optional
            The number of interpolation steps to process in one forward pass (default: 10).

        Returns
        -------
        torch.Tensor
            Gradients of shape (n_steps+1, B, C, H, W).
        """
        n_steps_plus_one, B, C, H, W = images.shape
        all_grads = []
        
        for start in range(0, n_steps_plus_one, chunk_size):
            end = min(start + chunk_size, n_steps_plus_one)
            # Flatten the (chunk_size, B) dimensions into one.
            chunk = images[start:end].reshape(-1, C, H, W).requires_grad_()
            
            outputs = self.model(chunk)
            outputs.sum().backward()
            
            grads = chunk.grad.detach().reshape(end - start, B, C, H, W)
            all_grads.append(grads)
            
            self.model.zero_grad()
        
        return torch.cat(all_grads, dim=0)

    def __integral_approximation(self, gradients: torch.Tensor) -> torch.Tensor:
        """
        Approximates the integral of the gradients using the Riemann Trapezoidal rule.
        
        Parameters
        ----------
        gradients : torch.Tensor
            Tensor of shape (n_steps+1, B, C, H, W) containing gradients along the path.

        Returns
        -------
        torch.Tensor
            Approximated integrated gradients of shape (B, C, H, W).
        """
        # Average adjacent gradients and then average over all interpolation steps.
        grads = (gradients[:-1] + gradients[1:]) / 2.0
        integrated_gradients = grads.mean(dim=0)
        return integrated_gradients

    def attribution_mask(self, image: torch.Tensor) -> np.ndarray:
        """
        Generates the Integrated Gradients attribution heatmap(s) using a fixed number of steps.
        
        Parameters
        ----------
        image : torch.Tensor
            Input tensor of shape (B, C, H, W) or (C, H, W). If necessary, a batch dimension is added.
        
        Returns
        -------
        np.ndarray
            Heatmap(s) as a numpy array of shape (B, H, W). For a single image input, the shape will be (1, H, W).
        """
        # Ensure image is batched.
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)
        
        baseline = torch.zeros_like(image).to(self.device)
        
        interpolated_images = self.__interpolate_image(baseline, image)
        gradients = self.__compute_gradients(interpolated_images)
        integrated_gradients = self.__integral_approximation(gradients)
        
        attribution = (image - baseline) * integrated_gradients
        
        heatmap = attribution.sum(dim=1)
        heatmap = torch.clamp(heatmap, min=0)
        heatmap = heatmap / (heatmap.amax(dim=(1, 2), keepdim=True) + 1e-7)
        
        return heatmap.detach().cpu().numpy()
