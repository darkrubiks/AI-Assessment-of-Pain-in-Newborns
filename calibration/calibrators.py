"""
calibrators.py

Author: Leonardo Antunes Ferreira
Date: 08/05/2023

This file contains calibrators for classification models.
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression

from calibration.metrics import negative_log_likelihood


def softmax(logit: np.ndarray) -> np.ndarray:
    """
    Returns the logit of a neural network as softmaxes probabilities.
    Prefer this when assessing multiclass problems.

    Parameters
    ----------
    logit : the output of the model in logit form

    Returns
    -------
    the softmax scores ranging from [0-1]
    """
    e_x = np.exp(logit - np.max(logit))
    return e_x / e_x.sum(axis=1, keepdims=1)


def sigmoid(logit: np.ndarray) -> np.ndarray:
    """
    Returns the logit of a neural network as a sigmoid probability.
    Prefer this when assessing binary problems.

    Parameters
    ----------
    logit : the output of the model in logit form

    Returns
    -------
    the sigmoid probability ranging from [0-1]
    """
    return 1 / (1 + np.exp(-logit))


class TemperatureScaling:
    """
    Scales the neural network logit with a value 1 / T (temperature), and returns
    the calibrated predictions. For now it only supports binary classification.

    Parameters
    ----------
    temperature : the temperature value T to scale the logit, must be > 0

    See Also
    -------
    Guo, Chuan, et al. "On calibration of modern neural networks."
    International conference on machine learning. PMLR, 2017.

    doi : https://doi.org/10.48550/arXiv.1706.04599
    """

    def __init__(self) -> None:
        self.T = None

    def _get_logits(self, probs: np.ndarray) -> np.ndarray:
        return np.log(probs / (1 - (probs-1e-5)))

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> None:
        """
        Minimizes the Negative Log Likelihood of the labels and logit
        to find the best temperature value.

        Parameters
        ----------
        probs : the output of the model in probability form. Only supports
        binary problems, provide it with the "True/Positive" class probability

        labels : the true labels as binary targets
        """
        logits = self._get_logits(probs)

        def _objective(T):
            return negative_log_likelihood(sigmoid(logits / T), labels)

        T = np.array([1.0])
        result = minimize(_objective, T, method='Nelder-Mead').x

        self.T = result[0]

    def predict(self, probs: np.ndarray) -> np.ndarray:
        """
        Returns the calibrated probabilities.

        Parameters
        ----------
        probs : the output of the model in probability form. Only supports
        binary problems, provide it with the "True/Positive" class probability

        Returns
        -------
        the calibrated probabilities ranging from [0-1]
        """
        logits = self._get_logits(probs)

        calibrated_probs = sigmoid(logits / self.T)

        return calibrated_probs


class IsotonicRegressor:
    """
    Fits a Isotonic Regressor to the probabilities and original labels, returns
    the calibrated probabilities. For this implementation it is preferred to use
    the "True/Positive" class.

    See Also
    --------
    Zadrozny, Bianca, and Charles Elkan. "Transforming classifier scores into
    accurate multiclass probability estimates." Proceedings of the eighth ACM
    SIGKDD international conference on Knowledge discovery and data mining. 2002.

    doi : https://doi.org/10.1145/775047.775151
    """

    def __init__(self) -> None:
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.function = None

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> None:
        """
        Fits the Isotonic Regressor using the available data.

        Parameters
        ----------
        probs : the output of the model in probability form. Only supports
        binary problems, provide it with the "True/Positive" class probability

        labels : the true labels as binary targets
        """
        self.calibrator.fit(probs.reshape(-1, 1), labels)
        self.function = self.calibrator.f_

    def predict(self, probs: np.ndarray) -> np.ndarray:
        """
        Returns the calibrated probabilities.

        Parameters
        ----------
        probs : the output of the model in probability form. Only supports
        binary problems, provide it with the "True/Positive" class probability

        Returns
        -------
        the calibrated probabilities ranging from [0-1]
        """
        calibrated_probs = self.calibrator.predict(probs)

        return calibrated_probs


class PlattScaling:
    """
    Implementation of the Platt Scalling method for calibrating probabilities.
    For this implementation it is preferred to use the "True/Positive" class.

    See Also
    --------
    Platt, John. "Probabilistic outputs for support vector machines and comparisons
    to regularized likelihood methods." Advances in large margin classifiers. 1999.
    """

    def __init__(self) -> None:
        self.A = None
        self.B = None

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> None:
        """
        Minimizes the Negative Log Likelihood to find the parameters A and B
        of a sigmoid that best calibrates the probabilities.

        Parameters
        ----------
        probs : the output of the model in probability form. Only supports
        binary problems, provide it with the "True/Positive" class probability

        labels : the true labels as binary targets
        """
        # Calculate priors according to Platt 1999.
        mask_negative_samples = labels <= 0
        N_minus = float(np.sum(mask_negative_samples))
        N_plus = labels.shape[0] - N_minus

        # The priors are used to make a soft label and helps prevent overfitting
        T = np.zeros_like(labels, dtype=np.float64)
        T[labels > 0] = (N_plus + 1.0) / (N_plus + 2.0)
        T[labels <= 0] = 1.0 / (N_minus + 2.0)

        def _objective(AB):
            P = 1 / (1 + np.exp((AB[0] * probs + AB[1])))
            return negative_log_likelihood(P, T)

        AB0 = np.array([0.0, np.log((N_minus + 1.0) / (N_plus + 1.0))])
        result = minimize(_objective, AB0, method='Nelder-Mead').x

        self.A = result[0]
        self.B = result[1]

    def predict(self, probs: np.ndarray) -> np.ndarray:
        """
        Returns the calibrated probabilities.

        Parameters
        ----------
        probs : the output of the model in probability form. Only supports
        binary problems, provide it with the "True/Positive" class probability

        Returns
        -------
        the calibrated probabilities ranging from [0-1]
        """
        calibrated_probs = 1 / (1 + np.exp((self.A * probs + self.B)))

        return calibrated_probs


class HistogramBinning:
    """
    Histogram Binning to calibrate the models predictions. Very simple and fast algorithm,
    it involves partitioning predicted probabilities into bins, calculating observed true
    frequency of events in each bin and adjusting predicted probabilities that fall in these
    bins.

    See Also
    --------
    Zadrozny, Bianca and Elkan, Charles. "Obtaining calibrated probability estimates from
    decision trees and naive bayesian classifiers". In ICML. 2001.
    """

    def __init__(self) -> None:
        self.bins = None
        self.prob_true = None

    def fit(self, probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> None:
        """
        Calculate the observed true frequency of envents in each bin. That later will be used
        to predict new calibrated probabilites.

        Parameters
        ----------
        probs : the output of the model in probability form. Only supports
        binary problems, provide it with the "True/Positive" class probability

        labels : the true labels as binary targets

        n_bins : number of bins to discretize
        """
        quantiles = np.linspace(0, 1, n_bins + 1)
        self.bins = np.percentile(probs, quantiles * 100)
        binids = np.searchsorted(self.bins[1:-1], probs)

        bin_true = np.bincount(binids, weights=labels, minlength=len(self.bins))
        bin_total = np.bincount(binids, minlength=len(self.bins))

        nonzero = bin_total != 0
        self.prob_true = bin_true[nonzero] / bin_total[nonzero]

    def predict(self, probs: np.ndarray) -> np.ndarray:
        """
        Returns the calibrated probabilities.

        Parameters
        ----------
        probs : the output of the model in probability form. Only supports
        binary problems, provide it with the "True/Positive" class probability

        Returns
        -------
        the calibrated probabilities ranging from [0-1]
        """
        binids = np.searchsorted(self.bins[1:-1], probs)
        calibrated_probs = self.prob_true[binids]

        return calibrated_probs