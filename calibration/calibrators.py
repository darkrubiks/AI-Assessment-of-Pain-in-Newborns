"""
calibrators.py

Author: Leonardo Antunes Ferreira
Date: 08/05/2023

This file contains calibrators for classification models.
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression


def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Returns the logits of a neural network as softmaxes probabilities.
    """
    e_x = np.exp(logits - np.max(logits))
    return e_x / e_x.sum(axis=1, keepdims=1)

class TemperatureScaling:
    """
    Scales the neural network logits with a value 1 / T (temperature), and returns
    the calibrated predictions.
    """
    def __init__(self, 
                 temperature: float=1.0) -> None:
        self.temperature = temperature

    def predict(self, 
                logits: np.ndarray) -> np.ndarray:
        calibrated_probs = softmax(logits / self.temperature)
        return calibrated_probs
    
class Isotonic:
    """
    Fits a Isotonic Regressor to the probabilities and original labels, returns the
    calibrated probabilities. This implementation only works for 2 classes, is
    preferred to use the "True/Positive" class.
    """
    def __init__(self, class_idx: int=1) -> None:
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.class_idx = class_idx
        
    def fit(self, 
            logits: np.ndarray, 
            y_true: np.ndarray) -> None:
        # Choose 70% of the data so the Regressor is not overfitted
        random_idx = np.random.randint(0, len(logits), int(0.7*len(logits)))
        probs = softmax(logits[random_idx])[:, self.class_idx]
        y_true = y_true[random_idx]
        self.calibrator.fit(probs, y_true)

    def predict(self, 
                logits: np.ndarray) -> np.ndarray:
        probs = softmax(logits)[:, self.class_idx]
        pred_probs = self.calibrator.predict(probs)
        calibrated_probs = np.zeros((len(probs), 2))
        calibrated_probs[:, 0] = 1 - pred_probs
        calibrated_probs[:, 1] = pred_probs
        return calibrated_probs