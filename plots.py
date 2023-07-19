import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from calibration.metrics import calibration_curve

matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False


def plot_calibration_curve(confs: np.ndarray, 
                           labels: np.ndarray, 
                           n_bins: int=10, 
                           plot_samples: bool=False) -> None:
    """
    Plots the calibration curve. It is also possible to include the amount
    of samples in each bin as a circle with dynamic radius.

    Parameters
    ----------
    confs : confidence on the positive class

    labels : true labels as binary targets

    n_bins : number of bins to discretize

    plot_samples : if True it will also plot the amount of samples in each
    bin like a circle where a bigger radius means more samples.
    """
    prob_true, prob_pred, bin_samples =  calibration_curve(confs, labels, n_bins)

    if plot_samples:
        total_bins = bin_samples.sum()

        for i, bin in enumerate(bin_samples):
            plt.plot(prob_pred[i], prob_true[i], marker='o', color='#c1272d', markersize=int((bin/total_bins)*100))

    plt.plot(prob_pred, prob_true,  linestyle='-', marker='o', color='#c1272d')
    plt.plot(np.arange(0,1.1,0.1), np.arange(0,1.1,0.1), 'k--')
    plt.title('Calibration Curve')
    plt.xlabel('Mean Predicted Confidence')
    plt.ylabel('Fraction of Positives')

    


