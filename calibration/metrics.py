"""
metrics.py

Author: Leonardo Antunes Ferreira
Date: 06/05/2023

This file contains calibration metrics like ECE and MCE. For more information see:

Naeini, Mahdi Pakdaman, Gregory Cooper, and Milos Hauskrecht. 
"Obtaining well calibrated probabilities using bayesian binning." 
Proceedings of the AAAI conference on artificial intelligence. 2015.

doi: https://doi.org/10.1609/aaai.v29i1.9602
"""
import numpy as np
from calibration.calibrators import softmax


def ECE(logits: np.ndarray,
        labels: np.ndarray,
        n_bins: int=10) -> np.float32:
    """
    Calculates the Expected Calibration Error of a model.

    The input to this metric is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    softmaxes = softmax(logits)
    predictions = np.argmax(softmaxes, axis=1)
    confidences = softmaxes[np.arange(len(predictions)), predictions]
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

def MCE(logits: np.ndarray,
        labels: np.ndarray,
        n_bins: int=10) -> np.float32:
    """
    Calculates the Maximum Calibration Error of a model.

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return the maximum gap.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    softmaxes = softmax(logits)
    predictions = np.argmax(softmaxes, axis=1)
    confidences = softmaxes[np.arange(len(predictions)), predictions]
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