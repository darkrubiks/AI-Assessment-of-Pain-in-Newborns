"""
metrics.py

Author: Leonardo Antunes Ferreira
Date: 06/05/2023

This file contains metrics that can be used to validate the model's calibration.
"""
from typing import Tuple

import numpy as np
from scipy.special import xlogy
from sklearn.metrics import brier_score_loss


def _bin_data(probs: np.ndarray,
              labels: np.ndarray,
              n_bins: int=10,
              mode: str='uniform') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bins the probabilities and labels.
    """
    if mode == 'uniform':
        # Equal width bins
        bins = np.linspace(0.0, 1.0, n_bins + 1)
    elif mode == 'quantile':
        # Same number of samples in each bin
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(probs, quantiles * 100)

    binids = np.searchsorted(bins[1:-1], probs)

    bin_sums = np.bincount(binids, weights=probs, minlength=len(bins))
    bin_true = np.bincount(binids, weights=labels, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]

    return  (prob_true, prob_pred, bin_total[nonzero])


def ECE(probs: np.ndarray,
        labels: np.ndarray,
        n_bins: int=10,
        mode: str='uniform') -> np.float32:
    """
    Calculates the Expected Calibration Error of a model.

    Parameters
    ----------
    probs : probability of the positive class

    labels : true labels as binary targets

    n_bins : number of bins to discretize

    mode : "uniform" for equal width bins or "quantile" for 
    equal amount of samples in bins

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
    prob_true, prob_pred, bin_total = _bin_data(probs, labels, n_bins, mode)

    P =  bin_total/np.sum(bin_total)
    ece = np.sum(np.abs(prob_true - prob_pred) * P)

    return ece


def MCE(probs: np.ndarray,
        labels: np.ndarray,
        n_bins: int=10,
        mode: str='uniform') -> np.float32:
    """
    Calculates the Maximum Calibration Error of a model.

    Parameters
    ----------
    probs : probability of the positive class

    labels : true labels as binary targets

    n_bins : number of bins to discretize

    mode : "uniform" for equal width bins or "quantile" for 
    equal amount of samples in bins

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
    prob_true, prob_pred, _ = _bin_data(probs, labels, n_bins, mode)

    mce = np.max(np.abs(prob_true - prob_pred))

    return mce

def negative_log_likelihood(probs: np.ndarray,
                            labels: np.ndarray) -> np.float32:
    """
    Calculates the Negative Log Likelihood for a binary problem.

    Parameters
    ----------
    probs : probability of the positive class

    labels : true labels as binary targets

    Returns
    -------
    nll : the Negative Log Likelihood
    """
    eps = np.finfo(probs.dtype).eps
    probs = np.clip(probs, eps, 1 - eps)
    nll = -(xlogy(labels, probs) + xlogy(1 - labels, 1 - probs))

    return nll.mean()


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
                      n_bins: int=10,
                      mode: str='uniform') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute true and predicted probabilities for a calibration curve.

    Adapted from scikitlearn to return the amount of positive class 
    samples in each bin.

    Parameters
    ----------
    probs : probability of the positive class

    labels : true labels as binary targets

    n_bins : number of bins to discretize

    mode : "uniform" for equal width bins or "quantile" for 
    equal amount of samples in bins

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
    prob_true, prob_pred, bin_total = _bin_data(probs, labels, n_bins, mode)
    
    return prob_true, prob_pred, bin_total