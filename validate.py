"""
validate.py

Author: Leonardo Antunes Ferreira
Date: 12/07/2022

Code for validating Deep Learning models.
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def validation_metrics(labels: np.ndarray,
                       preds: np.ndarray) -> dict:
    """
    Returns a dictionart with the following metrics: Accuracy,
    F1 Socre, Precision, Recall.

    Parameters
    ----------
    labels : the original labels for each sample
    preds : the predicted class for each sample

    Returns
    -------
    metrics : a dictionary containing the metrics
    """
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)

    metrics = {'Accuracy': acc, 'F1 Score': f1, 'Precision': precision, 'Recall': recall}

    return metrics