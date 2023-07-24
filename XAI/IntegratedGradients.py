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
    Integrated Gradients method.

    Parameters
    ----------
    model : the model to be used

    Returns
    -------
    heatmap : the attribution mask

    device : use CPU or CUDA device

    See Also
    --------
    Sundararajan, Mukund, Ankur Taly, and Qiqi Yan. "Axiomatic attribution for 
    deep networks." International conference on machine learning. PMLR, 2017.
    
    doi: https://doi.org/10.48550/arXiv.1703.01365
    """
    def __init__(self, 
                 model: torch.nn.Module,
                 device: str='cpu') -> None:
        self.device = device
        self.model = model.eval().to(self.device)
        
    def __interpolate_image(self,
                            baseline: torch.Tensor,
                            image: torch.Tensor,
                            n_steps: int) -> torch.Tensor:
        """
        Generates a linear interpolation between the baseline and the original
        image.
        """
        alphas = torch.linspace(start=0.0, end=1.0, steps=n_steps+1).to(self.device)
        alphas_x = alphas[..., None, None, None]

        baseline_x = torch.unsqueeze(baseline, 0)
        image_x = torch.unsqueeze(image, 0)

        delta = image_x - baseline_x

        interpoleted = baseline_x + alphas_x * delta

        return interpoleted

    def __compute_gradients(self,
                            image: torch.Tensor,
                            target_class: int) -> torch.Tensor:
        """
        Calculates gradients in order to measure the relationship between changes
        to a feature and changes in the model's predictions.
        """
        image = image.requires_grad_()
        logit = self.model(image)[:, target_class]
        logit.sum().backward()

        return image.grad

    def __integral_approximation(self,
                                 gradients: torch.Tensor) -> torch.Tensor:
        """
        Computes the numerical approximation of an integral for Integrated
        Gradients using the Riemann Trapezoidal rule.
        """
        grads = (gradients[:-1] + gradients[1:]) / 2.0
        integrated_gradients = grads.mean(dim=0)

        return integrated_gradients

    def attribution_mask(self,
                         image: torch.Tensor,
                         target_class: int,
                         n_steps: int=50) -> np.ndarray:
        """
        Generates the Integrated Gradients feature attributions.
        """
        image = image.to(self.device)
        baseline = torch.zeros(image.size()).to(self.device)

        interpoleted = self.__interpolate_image(baseline, image, n_steps)
        gradients = self.__compute_gradients(interpoleted, target_class)
        integrated_gradients = self.__integral_approximation(gradients)
        
        attribution = (image - baseline) * integrated_gradients

        heatmap = torch.abs(attribution.sum(dim=0))
        heatmap = heatmap.detach().cpu().numpy().squeeze()

        return heatmap