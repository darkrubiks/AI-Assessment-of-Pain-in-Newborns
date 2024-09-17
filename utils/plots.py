import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from calibration.metrics import ECE, calibration_curve

def plot_calibration_curve(probs: np.ndarray, 
                           labels: np.ndarray, 
                           n_bins: int=10,
                           mode: str='uniform',
                           path: str=os.getcwd()) -> None:
    """
    Plots the calibration curve. It also includes the predictions probabilities
    histogram.

    Parameters
    ----------
    probs : probability of the positive class

    labels : true labels as binary targets

    n_bins : number of bins to discretize

    mode : "uniform" for equal width bins or "quantile" for 
    equal amount of samples in bins

    path : where to save the plot image. Defaults to current directory.
    """
    fig = plt.figure()
    gs = fig.add_gridspec(2, 1,  height_ratios=(3, 1), left=0.1, right=0.9, bottom=0.1, top=0.9, hspace=0.02)

    ax_curve = fig.add_subplot(gs[0])
    ax_hist = fig.add_subplot(gs[1], sharex=ax_curve)
    
    prob_true, prob_pred, _ =  calibration_curve(probs, labels, n_bins, mode)
    
    ece = ECE(probs, labels, n_bins, mode)

    ax_curve.plot(np.arange(0,1.1,0.1), np.arange(0,1.1,0.1), linestyle=":", color='#9e9e9e')
    ax_curve.plot(prob_pred, prob_true,  linestyle='-', marker='o', label=f'ECE = {ece:.4f}')
    ax_hist.hist(probs, bins=n_bins, edgecolor='k')

    ax_curve.set_title('Calibration Curve')
    ax_curve.set_ylabel('Fraction of Positives')
    ax_curve.legend()

    ax_hist.set_xlabel('Mean Predicted Probability')
    ax_hist.set_ylabel('Count')

    plt.savefig(os.path.join(path,'calibration_curve.png'), 
                dpi=300, 
                bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(preds: np.ndarray, 
                          labels: np. ndarray,
                          classes: List[str],
                          path: str=os.getcwd()) -> None:
    """
    Plots the Confusion Matrix.

    Parameters
    ----------
    preds : model's predictions

    labels : true labels as binary targets

    classes : a list of class names to use. The class names order 
    should exactly match the ordinality of the labels and predictions.

    path : where to save the plot image. Defaults to current directory.
    """
    cm = confusion_matrix(labels, preds)

    plt.imshow(cm, cmap=plt.cm.Blues)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes, rotation=90)

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                     horizontalalignment='center',
                     color='white' if cm[i, j] > thresh else 'black')
            
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(path,'confusion_matrix.png'), 
                dpi=300, 
                bbox_inches='tight')
    plt.close()


def plot_roc_curve(probs: np.ndarray, 
                   labels: np.ndarray,
                   path: str=os.getcwd()) -> None:
    """
    Plots the Receiver Operating Characteristic curve.

    Parameters
    ----------
    probs : probability of the positive class

    labels : true labels as binary targets

    path : where to save the plot image. Defaults to current directory.
    """
    fpr, tpr, _ = roc_curve(labels, probs)

    auc = roc_auc_score(labels, probs)

    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.title(f'ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Postive Rate')
    plt.legend()
    plt.savefig(os.path.join(path,'roc_curve.png'), 
                dpi=300, 
                bbox_inches='tight')
    plt.close()


def plot_pre_rec_curve(probs: np.ndarray,
                       labels: np.ndarray,
                       path: str=os.getcwd()) -> None:
    """
    Plots the Precision-Recall curve.

    Parameters
    ----------
    probs : probability of the positive class

    labels : true labels as binary targets

    path : where to save the plot image. Defaults to current directory.
    """
    precision, recall, _ = precision_recall_curve(labels, probs)

    ap = average_precision_score(labels, probs)
    
    plt.plot(recall, precision, label=f'AP = {ap:.4f}')
    plt.title(f'Precision-Recall Curve ')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig(os.path.join(path,'precision_recall_curve.png'), 
                dpi=300, 
                bbox_inches='tight')
    plt.close()


def plot_results_above_threshold(probs: np.ndarray,
                                 labels: np.ndarray,
                                 path: str=os.getcwd()) -> None:
    """
    Plots the Accuracy, Precision, Recall and F1 Score results
    by changing the probability threshold.

    Parameters
    ----------
    probs : probability of the positive class

    labels : true labels as binary targets

    path : where to save the plot image. Defaults to current directory.
    """
    metrics = {'Percentage of Samples': '', 
               'Accuracy': accuracy_score, 
               'Precision': precision_score, 
               'Recall': recall_score,
               'F1 Score': f1_score}

    threshold = np.arange(0.01, 1.00, 0.01)

    for key in metrics.keys():
        above_threshold = np.zeros(len(threshold))

        for i,thr in enumerate(threshold):
            preds = (probs > thr).astype(int)
            if key == 'Percentage of Samples':
                above_threshold[i] = sum(preds)/len(probs)
            elif key == 'Accuracy':
                above_threshold[i] = metrics[key](labels, preds)
            else:
                above_threshold[i] = metrics[key](labels, preds, zero_division=0)

            if thr == 0.5:
                result = above_threshold[i]

        nonzero = np.array(above_threshold) != 0
        plt.figure()
        plt.plot(threshold[nonzero], above_threshold[nonzero], label=f'{key} at 0.5 = {result:.4f}')
        plt.xlabel('Probability Threshold')
        plt.ylabel(key)
        plt.legend()
        plt.savefig(os.path.join(path,f'{key}.png'), 
                    dpi=300, 
                    bbox_inches='tight')
        plt.close()


def probability_histogram(probs: np.ndarray, 
                          labels: np.ndarray, 
                          bins: int=10, 
                          threshold: float=0.5, 
                          path: str=os.getcwd()) -> None:
    """
    Plots an histogram containing the predicted probabilities for each
    confusion matrix category TN, TP, FN and FP. Also plots the NPV (Negative
    Predicted Value) and PPV (Positive Predicted Value).

    Parameters
    ----------
    probs : probability of the positive class

    labels : true labels as binary targets

    bins : the number of bins to discretize the probabilities

    threshold : the probability threshold to consider as a positive prediction

    path : where to save the plot image. Defaults to current directory
    """
    preds = (probs > threshold).astype(int)

    bins_P = np.linspace(0.5, 1.0, bins + 1) # Positive values bins
    bins_N = np.linspace(0.0, 0.5, bins + 1) # Negative values bins

    probs_TN = probs[(labels == preds) & (labels == 0)]
    probs_TP = probs[(labels == preds) & (labels == 1)]
    probs_FN = probs[(labels != preds) & (labels == 1)]
    probs_FP = probs[(labels != preds) & (labels == 0)]

    # Create plot
    fig = plt.figure(figsize=(15,5))
    gs = fig.add_gridspec(
        2, 1,  
        height_ratios=(4, 2), 
        left=0.1, 
        right=0.9, 
        bottom=0.1, 
        top=0.9, 
        hspace=0.2
    )

    ax_curve = fig.add_subplot(gs[1])
    ax_hist = fig.add_subplot(gs[0])

    all_bin_sums = []
    y_max = 0
    # Iterate over predictions and plot the histogram
    for probs, bins, label in zip(
        [probs_TN, probs_TP, probs_FN, probs_FP],
        [bins_N, bins_P, bins_N, bins_P],
        ['VN', 'VP', 'FN', 'FP']
        ):

        bin_sums = []

        binids = np.searchsorted(bins[1:-1], probs)
        bin_sums.append(np.bincount(binids, minlength=len(bins)))
        bin_sums = np.array(bin_sums)

        all_bin_sums.append(bin_sums.ravel())
    
        ax_hist.bar(
            bins, 
            bin_sums.mean(axis=0), 
            width=(bins_P[1]-bins_P[0]), 
            align='edge', 
            edgecolor='k', 
            alpha=0.8, 
            label=label
        )
        
        # Get maximum values to plot vertical lines below
        if bin_sums.mean(axis=0).max() > y_max:
            y_max = bin_sums.mean(axis=0).max()

        ax_hist.vlines(
            x=probs.mean(), 
            ymin=0, 
            ymax=y_max + 10, 
            color='#9e9e9e', 
            linestyle=':'
        )
        
        ax_hist.text(
            x=probs.mean() - 0.02, 
            y=y_max + 15, 
            s=f'${label}$={probs.mean():.2f}', 
            verticalalignment='center', 
            rotation='vertical'
        )

    # Plot NPV and PPV values
    # The X points will be centered with respective to the histogram bar above
    # The Y values are found by using the previously created variable all_bim_sums 
    # that contains the TN, TP, FN and FP quantities in each bin
    ax_curve.plot(
        bins_N + (bins_P[1] - bins_P[0]) / 2, 
        (all_bin_sums[0] / (all_bin_sums[0] + all_bin_sums[2])), 
        's-', 
        label='VNP'
    )
    
    ax_curve.plot(
        bins_P + (bins_P[1] - bins_P[0]) / 2, 
        (all_bin_sums[1] / (all_bin_sums[1] + all_bin_sums[3])), 
        'o-', 
        label='VPP'
    )
    
    ax_curve.hlines(
        y=0.5, 
        xmin=0, 
        xmax=1, 
        color='#9e9e9e',
        linestyle=':'
    )

    ax_hist.set_xlim([-0.05, 1.05])
    ax_hist.set_xticklabels([])

    ax_curve.set_xlim([-0.05, 1.05])
    ax_curve.set_ylim([-0, 1.05])

    ax_hist.set_ylabel('Quantidade')
    ax_hist.legend(loc='upper right')

    ax_curve.set_xlabel('Probabilidade Prevista Média')
    ax_curve.set_ylabel('Proporção \%')
    ax_curve.legend(loc='upper right')

    plt.savefig(
        os.path.join(path,f'hist.pdf'), 
        dpi=300, 
        bbox_inches='tight'
    )
    plt.show()
    plt.close()
