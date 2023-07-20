from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score

from calibration.metrics import ECE, calibration_curve

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
    bin like a circle where a bigger radius means more samples
    """
    prob_true, prob_pred, bin_samples =  calibration_curve(confs, labels, n_bins)

    ece = ECE(confs, labels, n_bins)

    if plot_samples:
        total_bins = bin_samples.sum()
        for i, bin in enumerate(bin_samples):
            plt.plot(prob_pred[i], prob_true[i], marker='o', color='#c1272d', markersize=int((bin/total_bins)*100))

    plt.plot(prob_pred, prob_true,  linestyle='-', marker='o', color='#c1272d')
    plt.plot(np.arange(0,1.1,0.1), np.arange(0,1.1,0.1), 'k--')
    plt.title(f'Calibration Curve - ECE = {ece:.4f}')
    plt.xlabel('Mean Predicted Confidence')
    plt.ylabel('Fraction of Positives')
    plt.savefig('calibration_curve.png', dpi=300, bbox_inches='tight')


def plot_confusion_matrix(preds: np.ndarray, 
                          labels: np. ndarray,
                          classes: List[str]) -> None:
    """
    Plots the Confusion Matrix.

    Parameters
    ----------
    preds : model's predictions

    labels : true labels as binary targets

    classes : a list of class names to use. The class names order 
    should exactly match the ordinality of the labels and predictions.
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
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')


def plot_roc_curve(confs: np.ndarray, 
                   labels: np.ndarray) -> None:
    """
    Plots the Receiver Operating Characteristic curve.

    Parameters
    ----------
    confs : confidence on the positive class

    labels : true labels as binary targets
    """
    fpr, tpr, _ = roc_curve(labels, confs)

    auc = roc_auc_score(labels, confs)

    plt.plot(fpr, tpr)
    plt.title(f'ROC Curve - AUC = {auc:.4f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Postive Rate')
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')


def plot_pre_rec_curve(confs: np.ndarray,
                       labels: np.ndarray) -> None:
    """
    Plots the Precision-Recall curve.

    Parameters
    ----------
    confs : confidence on the positive class

    labels : true labels as binary targets
    """
    precision, recall, _ = precision_recall_curve(labels, confs)

    ap = average_precision_score(labels, confs)
    
    plt.plot(recall, precision)
    plt.title(f'Precision-Recall Curve - AP = {ap:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig('precision_recall_curve.png', dpi=300, bbox_inches='tight')