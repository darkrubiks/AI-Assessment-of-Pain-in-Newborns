"""
metrics.py

Author: Leonardo Antunes Ferreira
Date: 06/05/2023

This file contains metrics that can be used to validate the model's calibration.
"""
import numpy as np
from sklearn.metrics import log_loss, brier_score_loss


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

    labels : true labels as binary targets

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

def negative_log_likelihood(confs: np.ndarray,
                            labels: np.ndarray) -> np.float32:
    """
    Calculates the Negative Log Likelihood.

    Parameters
    ----------
    confs : confidence on the positive class

    labels : true labels as binary targets

    Returns
    -------
    nll : the Negative Log Likelihood

    See Also
    --------
    scikitlearn : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html
    """
    nll = log_loss(labels, confs)

    return nll

def brier_score(confs: np.ndarray, 
                labels: np.ndarray) -> np.float32:
    """
    Calculates the Brier Score.

    Parameters
    ----------
    confs : confidence on the positive class

    labels : true labels as binary targets

    Returns
    -------
    brier : the Negative Log Likelihood

    See Also
    --------
    scikitlearn : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html
    """
    brier = brier_score_loss(labels, confs)

    return brier

def calibration_curve(confs: np.ndarray,
                      labels: np.ndarray,
                      n_bins: int=10) -> np.ndarray:
    """
    Compute true and predicted probabilities for a calibration curve.

    Adapted from scikitlearn to return the amount of positive class 
    samples in each bin.

    Parameters
    ----------
    confs : confidence on the positive class

    labels : true labels as binary targets

    n_bins : number of bins to discretize

    Returns
    -------
    prob_true : the proportion of samples whose class is the positive class,
    in eachbin (fraction of positives)

    prob_pred : the mean predicted probability in each bin

    bin_samples : the amount of samples in each bin

    See Also
    --------
    scikitlearn : https://scikit-learn.org/stable/modules/generated/sklearn.calibration.calibration_curve.html
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.searchsorted(bins[1:-1], confs)
    bin_samples = np.bincount(binids)

    bin_sums = np.bincount(binids, weights=confs, minlength=len(bins))
    bin_true = np.bincount(binids, weights=labels, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]
    
    return prob_true, prob_pred, bin_samples