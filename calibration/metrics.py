"""
metrics.py

Author: Leonardo Antunes Ferreira
Date: 06/05/2023

This file contains metrics that can be used to validate the model's calibration.
"""
import numpy as np


def ECE(probs: np.ndarray,
        labels: np.ndarray,
        n_bins: int=10) -> np.float32:
    """
    Calculates the Expected Calibration Error of a model.

    Parameters
    ----------
    probs : the predicted softmax scores of both classes

    labels : the true labels as binary targetes

    n_bins : the number of bins to discretize

    Returns
    -------
    ece : the Expected Calibration Error

    See Also
    --------
    Naeini, Mahdi Pakdaman, Gregory Cooper, and Milos Hauskrecht. 
    "Obtaining well calibrated probabilities using bayesian binning." 
    Proceedings of the AAAI conference on artificial intelligence. 2015.

    doi : https://doi.org/10.1609/aaai.v29i1.9602
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    predictions = np.argmax(probs, axis=1)
    confidences = probs[np.arange(len(predictions)), predictions]
    accuracies = np.equal(predictions, labels)

    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = np.greater(confidences, bin_lower) * np.less(confidences, bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece

def MCE(probs: np.ndarray,
        labels: np.ndarray,
        n_bins: int=10) -> np.float32:
    """
    Calculates the Maximum Calibration Error of a model.

    Parameters
    ----------
    probs : the predicted softmax scores of both classes

    labels : the true labels as binary targetes

    n_bins : the number of bins to discretize

    Returns
    -------
    ece : the Expected Calibration Error

    See Also
    --------
    Naeini, Mahdi Pakdaman, Gregory Cooper, and Milos Hauskrecht. 
    "Obtaining well calibrated probabilities using bayesian binning." 
    Proceedings of the AAAI conference on artificial intelligence. 2015.

    doi: https://doi.org/10.1609/aaai.v29i1.9602
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    predictions = np.argmax(probs, axis=1)
    confidences = probs[np.arange(len(predictions)), predictions]
    accuracies = np.equal(predictions, labels)

    mce = []
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = np.greater(confidences, bin_lower) * np.less(confidences, bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            mce.append(np.abs(avg_confidence_in_bin - accuracy_in_bin))

    return max(mce)