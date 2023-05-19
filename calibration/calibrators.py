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

    Parameters
    ----------
    logits : the output of the model in logit form (NOT the softmax), from both 
    classes

    Returns
    -------
    the softmax scores ranging from [0-1]
    """
    e_x = np.exp(logits - np.max(logits))
    return e_x / e_x.sum(axis=1, keepdims=1)

class TemperatureScaling:
    """
    Scales the neural network logits with a value 1 / T (temperature), and returns
    the calibrated predictions.

    Parameters
    ----------
    temperature : the temperature value T to scale the logits, must be > 1

    See Also
    -------
    Guo, Chuan, et al. "On calibration of modern neural networks." 
    International conference on machine learning. PMLR, 2017.

    doi : https://doi.org/10.48550/arXiv.1706.04599
    """
    def __init__(self, 
                 temperature: float=1.0) -> None:
        self.temperature = temperature

    def predict(self, 
                logits: np.ndarray) -> np.ndarray:
        """
        Returns the calibrated probabilities.

        Parameters
        ----------
        logits : the output of the model in logit form (NOT the softmax), from 
        both classes

        Returns
        -------
        the calibrated probabilities ranging from [0-1]
        """
        calibrated_probs = softmax(logits / self.temperature)
        return calibrated_probs
    
class Isotonic:
    """
    Fits a Isotonic Regressor to the probabilities and original labels, returns 
    the calibrated probabilities. This implementation only works for 2 classes,
    it is preferred to use the "True/Positive" class.

    Parameters
    ----------
    class_idx : "True/Positive" class index

    See Also
    --------
    Zadrozny, Bianca, and Charles Elkan. "Transforming classifier scores into 
    accurate multiclass probability estimates." Proceedings of the eighth ACM 
    SIGKDD international conference on Knowledge discovery and data mining. 2002.

    doi : https://doi.org/10.1145/775047.775151
    """
    def __init__(self, class_idx: int=1) -> None:
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.class_idx = class_idx
        
    def fit(self, 
            logits: np.ndarray, 
            labels: np.ndarray) -> None:
        """
        Fits the Isotonic Regressor using the available data. It automatically
        splits into train and test sets so the Regressor won't overfit.

        Parameters
        ----------
        logits : the output of the model in logit form (NOT the softmax), from 
        both classes

        labels : the true labels as binary targetes
        """
        # Choose 70% of the data so the Regressor is not overfitted
        random_idx = np.random.randint(0, len(logits), int(0.7*len(logits)))
        probs = softmax(logits[random_idx])[:, self.class_idx]
        labels = labels[random_idx]
        self.calibrator.fit(probs, labels)

    def predict(self, 
                logits: np.ndarray) -> np.ndarray:
        """
        Returns the calibrated probabilities.

        Parameters
        ----------
        logits : the output of the model in logit form (NOT the softmax), from 
        both classes

        Returns
        -------
        the calibrated probabilities ranging from [0-1]
        """
        probs = softmax(logits)[:, self.class_idx]
        pred_probs = self.calibrator.predict(probs)
        calibrated_probs = np.zeros((len(probs), 2))
        calibrated_probs[:, 0] = 1 - pred_probs
        calibrated_probs[:, 1] = pred_probs
        return calibrated_probs