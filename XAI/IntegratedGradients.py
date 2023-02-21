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
import torch
import torch.nn.functional as F


class IntegratedGradients:
    def __init__(self, model, n_steps):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.eval().to(self.device)
        self.n_steps = n_steps
        
    def __interpolate_image(self, baseline, image):
        """
        Generates a linear interpolation between the baseline and the original
        image.
        """
        alphas = torch.linspace(start=0.0, end=1.0, steps=self.n_steps+1).to(self.device)
        alphas_x = alphas[..., None, None, None]

        baseline_x = torch.unsqueeze(baseline, 0)
        image_x = torch.unsqueeze(image, 0)

        delta = image_x - baseline_x

        interpoleted = baseline_x + alphas_x * delta

        return interpoleted

    def __compute_gradients(self, image, target_class):
        """
        Calculates gradients in order to measure the relationship between changes
        to a feature and changes in the model's predictions.
        """
        image = image.requires_grad_()
        logit = self.model(image)
        probs = F.softmax(logit, 1)[:, target_class]
        probs.sum().backward()

        return image.grad

    def __integral_approximation(self, gradients):
        """
        Computes the numerical approximation of an integral for Integrated
        Gradients using the Riemann Trapezoidal rule.
        """
        grads = (gradients[:-1] + gradients[1:]) / 2.0
        integrated_gradients = grads.mean(dim=0)

        return integrated_gradients

    def attribution_mask(self, baseline, image, target_class):
        """
        Generates the Integrated Gradients feature attributions.
        """
        image = image.to(self.device)
        baseline = baseline.to(self.device)

        interpoleted = self.__interpolate_image(baseline, image)
        gradients = self.__compute_gradients(interpoleted, target_class)
        integrated_gradients = self.__integral_approximation(gradients)
        
        attribution = (image - baseline) * integrated_gradients

        heatmap = torch.abs(attribution.sum(dim=0))

        return heatmap.detach().cpu()