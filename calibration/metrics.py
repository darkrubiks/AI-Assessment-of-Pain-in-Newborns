"""
metrics.py

Author: Leonardo Antunes Ferreira
Date: 06/05/2023

This file contains metrics that can be used to validate the model's calibration.
"""
import numpy as np


def ECE(confs: np.ndarray,
        labels: np.ndarray,
        n_bins: int=10) -> np.float32:
    """
    Calculates the Expected Calibration Error of a model.

    Parameters
    ----------
    probs : confidence on the positive class

    labels : true labels as binary targets

    n_bins : number of bins to discretize

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

    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculate |fraction of positives - confidence| in each bin
        in_bin = np.greater(confs, bin_lower) * np.less(confs, bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            fraction_of_postives_in_bin = labels[in_bin].mean()
            avg_confidence_in_bin = confs[in_bin].mean()
            ece += np.abs(fraction_of_postives_in_bin - avg_confidence_in_bin) * prop_in_bin

    return ece

def MCE(confs: np.ndarray,
        labels: np.ndarray,
        n_bins: int=10) -> np.float32:
    """
    Calculates the Maximum Calibration Error of a model.

    Parameters
    ----------
    confs : confidence on the positive class

    labels : true labels as binary targetes

    n_bins : number of bins to discretize

    Returns
    -------
    mce : the Maximum Calibration Error

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

    mce = []
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculate |fraction of positives - confidence| in each bin
        in_bin = np.greater(confs, bin_lower) * np.less(confs, bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            fraction_of_postives_in_bin = labels[in_bin].mean()
            avg_confidence_in_bin = confs[in_bin].mean()
            mce.append(np.abs(avg_confidence_in_bin - fraction_of_postives_in_bin))

    return max(mce)