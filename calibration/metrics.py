"""
metrics.py

Author: Leonardo Antunes Ferreira
Date: 06/05/2023

This file contains metrics that can be used to validate the model's calibration.
"""
import numpy as np
from sklearn.metrics import brier_score_loss, log_loss


def _bin_data(probs: np.ndarray,
              labels: np.ndarray,
              n_bins: int=10) -> np.ndarray:
    """
    Bins the probabilities and labels.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.searchsorted(bins[1:-1], probs)

    bin_sums = np.bincount(binids, weights=probs, minlength=len(bins))
    bin_true = np.bincount(binids, weights=labels, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]

    return  prob_true, prob_pred, bin_total[nonzero]


def ECE(probs: np.ndarray,
        labels: np.ndarray,
        n_bins: int=10) -> np.float32:
    """
    Calculates the Expected Calibration Error of a model.

    Parameters
    ----------
    probs : probability of the positive class

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
    prob_true, prob_pred, bin_total = _bin_data(probs, labels, n_bins)

    P =  bin_total/np.sum(bin_total)
    ece = np.sum(np.abs(prob_true - prob_pred) * P)

    return ece


def MCE(probs: np.ndarray,
        labels: np.ndarray,
        n_bins: int=10) -> np.float32:
    """
    Calculates the Maximum Calibration Error of a model.

    Parameters
    ----------
    probs : probability of the positive class

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
    prob_true, prob_pred, _ = _bin_data(probs, labels, n_bins)

    mce = np.max(np.abs(prob_true - prob_pred))

    return mce


def negative_log_likelihood(probs: np.ndarray,
                            labels: np.ndarray) -> np.float32:
    """
    Calculates the Negative Log Likelihood.

    Parameters
    ----------
    probs : probability of the positive class

    labels : true labels as binary targets

    Returns
    -------
    nll : the Negative Log Likelihood

    See Also
    --------
    scikitlearn : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html
    """
    nll = log_loss(labels, probs)

    return nll


def brier_score(probs: np.ndarray, 
                labels: np.ndarray) -> np.float32:
    """
    Calculates the Brier Score.

    Parameters
    ----------
    probs : probability of the positive class

    labels : true labels as binary targets

    Returns
    -------
    brier : the Negative Log Likelihood

    See Also
    --------
    scikitlearn : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html
    """
    brier = brier_score_loss(labels, probs)

    return brier


def calibration_curve(probs: np.ndarray,
                      labels: np.ndarray,
                      n_bins: int=10) -> np.ndarray:
    """
    Compute true and predicted probabilities for a calibration curve.

    Adapted from scikitlearn to return the amount of positive class 
    samples in each bin.

    Parameters
    ----------
    probs : probability of the positive class

    labels : true labels as binary targets

    n_bins : number of bins to discretize

    Returns
    -------
    prob_true : the proportion of samples whose class is the positive class,
    in eachbin (fraction of positives)

    prob_pred : the mean predicted probability in each bin

    bin_total : the amount of samples in each bin

    See Also
    --------
    scikitlearn : https://scikit-learn.org/stable/modules/generated/sklearn.calibration.calibration_curve.html
    """
    prob_true, prob_pred, bin_total = _bin_data(probs, labels, n_bins)
    
    return prob_true, prob_pred, bin_total