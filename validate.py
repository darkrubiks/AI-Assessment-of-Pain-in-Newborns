"""
validate.py

Author: Leonardo Antunes Ferreira
Date: 12/07/2022

Code for validating Deep Learning models.
"""

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

from calibration.metrics import ECE, MCE, brier_score, negative_log_likelihood
from utils.plots import *


def validation_metrics(preds: np.ndarray, 
                       probs: np.ndarray,
                       labels: np.ndarray) -> dict:
    """
    Returns a dictionary with the following metrics: Accuracy,
    F1 Score, Precision, Recall.

    Parameters
    ----------
    preds : the predicted class for each sample

    probs : the probability of the prediction

    labels : the original labels for each sample

    Returns
    -------
    metrics : a dictionary containing the metrics
    """
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0,1]).ravel()

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total != 0 else np.nan

    precision_den = tp + fp
    precision = tp / precision_den if precision_den != 0 else 0.0  # undefined => 0.0

    sensitivity_den = tp + fn
    sensitivity = tp / sensitivity_den if sensitivity_den != 0 else 0.0  # aka recall

    specificity_den = tn + fp
    specificity = tn / specificity_den if specificity_den != 0 else 0.0

    f1_den = 2 * tp + fp + fn
    f1 = (2 * tp) / f1_den if f1_den != 0 else 0.0

    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = np.nan  # e.g., only one class present in labels

    metrics = {
        "Accuracy": accuracy,
        "F1 Score": f1,
        "Precision": precision,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "AUC": auc,
    }

    return metrics


def calibration_metrics(probs: np.ndarray,
                        labels: np.ndarray,
                        n_bins: int=10,
                        mode: str='uniform') -> dict:
    """
    Returns a dictionary with the following metrics: ECE, MCE,
    NLL, and Brier Score.

    Parameters
    ----------
    probs : probability of the positive class

    labels : the original labels for each sample

    mode : "uniform" for equal width bins or "quantile" for
    equal amount of samples in bins. Used for ECE and MCE

    Returns
    -------
    metrics : a dictionary containing the metrics
    """
    ece = ECE(probs, labels, n_bins, mode)
    mce = MCE(probs, labels, n_bins, mode)
    nll = negative_log_likelihood(probs, labels)
    brier = brier_score(probs, labels)

    metrics = {"ECE": ece, 
               "MCE": mce, 
               "NLL": nll, 
               "Brier": brier}

    return metrics


def validation_plots(preds: np.ndarray,
                     probs: np.ndarray,
                     labels: np.ndarray,
                     mode: str='uniform',
                     path: str=os.getcwd()) -> None:
    """
    Creates all the available plots for validation.

    Parameters
    ----------
    preds : the predicted class for each sample

    probs : probability of the positive class

    labels : the original labels for each sample

    mode : "uniform" for equal width bins or "quantile" for
    equal amount of samples in bins. Used for calibration plot
    """
    plot_calibration_curve(probs, labels, mode=mode, path=path)
    plot_confusion_matrix(preds, labels, ["No Pain", "Pain"], path)
    plot_roc_curve(probs, labels, path)
    plot_pre_rec_curve(probs, labels, path)
    plot_results_above_threshold(probs, labels, path)
