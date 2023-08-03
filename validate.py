"""
validate.py

Author: Leonardo Antunes Ferreira
Date: 12/07/2022

Code for validating Deep Learning models.
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils.plots import *


def validation_metrics(preds: np.ndarray,
                       labels: np.ndarray) -> dict:
    """
    Returns a dictionary with the following metrics: Accuracy,
    F1 Score, Precision, Recall.

    Parameters
    ----------
    preds : the predicted class for each sample

    labels : the original labels for each sample

    Returns
    -------
    metrics : a dictionary containing the metrics
    """
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)

    metrics = {'Accuracy': acc, 'F1 Score': f1, 'Precision': precision, 'Recall': recall}

    return metrics


def validation_plots(preds: np.ndarray,
                     probs: np.ndarray,
                     labels: np.ndarray,
                     path: str=os.getcwd()) -> None:
    """
    Creates all the available plots for validation.

    Parameters
    ----------
    preds : the predicted class for each sample

    probs : probability of the positive class

    labels : the original labels for each sample
    """
    plot_calibration_curve(probs, labels, plot_samples=True, path=path)
    plot_confusion_matrix(preds, labels, ['No Pain', 'Pain'], path)
    plot_roc_curve(probs, labels, path)
    plot_pre_rec_curve(probs, labels, path)
    plot_results_above_threshold(probs, labels, path)