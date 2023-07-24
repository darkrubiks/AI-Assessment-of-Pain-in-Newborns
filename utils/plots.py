import os
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from calibration.metrics import ECE, calibration_curve

matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False

COLOR = '#303eba'


def plot_calibration_curve(confs: np.ndarray, 
                           labels: np.ndarray, 
                           n_bins: int=10, 
                           plot_samples: bool=False,
                           path: str=os.getcwd()) -> None:
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

    path : where to save the plot image. Defaults to current directory.
    """
    prob_true, prob_pred, bin_samples =  calibration_curve(confs, labels, n_bins)

    ece = ECE(confs, labels, n_bins)

    if plot_samples:
        total_bins = bin_samples.sum()
        for i, bin in enumerate(bin_samples):
            plt.plot(prob_pred[i], prob_true[i], 
                     marker='o', color=COLOR, markersize=int((bin/total_bins)*100))

    plt.plot(prob_pred, prob_true,  linestyle='-', marker='o', color=COLOR, label=f'ECE = {ece:.4f}')
    plt.plot(np.arange(0,1.1,0.1), np.arange(0,1.1,0.1), 'k--')
    plt.title(f'Calibration Curve')
    plt.xlabel('Mean Predicted Confidence')
    plt.ylabel('Fraction of Positives')
    plt.legend()
    plt.savefig(os.path.join(path,'calibration_curve.png'), 
                dpi=300, 
                bbox_inches='tight')


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


def plot_roc_curve(confs: np.ndarray, 
                   labels: np.ndarray,
                   path: str=os.getcwd()) -> None:
    """
    Plots the Receiver Operating Characteristic curve.

    Parameters
    ----------
    confs : confidence on the positive class

    labels : true labels as binary targets

    path : where to save the plot image. Defaults to current directory.
    """
    fpr, tpr, _ = roc_curve(labels, confs)

    auc = roc_auc_score(labels, confs)

    plt.plot(fpr, tpr, color=COLOR, label=f'AUC = {auc:.4f}')
    plt.title(f'ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Postive Rate')
    plt.legend()
    plt.savefig(os.path.join(path,'roc_curve.png'), 
                dpi=300, 
                bbox_inches='tight')


def plot_pre_rec_curve(confs: np.ndarray,
                       labels: np.ndarray,
                       path: str=os.getcwd()) -> None:
    """
    Plots the Precision-Recall curve.

    Parameters
    ----------
    confs : confidence on the positive class

    labels : true labels as binary targets

    path : where to save the plot image. Defaults to current directory.
    """
    precision, recall, _ = precision_recall_curve(labels, confs)

    ap = average_precision_score(labels, confs)
    
    plt.plot(recall, precision, color=COLOR, label=f'AP = {ap:.4f}')
    plt.title(f'Precision-Recall Curve ')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig(os.path.join(path,'precision_recall_curve.png'), 
                dpi=300, 
                bbox_inches='tight')


def plot_results_above_threshold(confs: np.ndarray,
                                 labels: np.ndarray,
                                 path: str=os.getcwd()) -> None:
    """
    Plots the Accuracy, Precision, Recall and F1 Score results
    by changing the confidence threshold.

    Parameters
    ----------
    confs : confidence on the positive class

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
            preds = (confs > thr).astype(int)
            if key == 'Percentage of Samples':
                above_threshold[i] = sum(preds)/len(confs)
            else:
                above_threshold[i] = metrics[key](labels, preds)

        nonzero = np.array(above_threshold) != 0
        plt.figure()
        plt.plot(threshold[nonzero], above_threshold[nonzero], color=COLOR)
        plt.xlabel('Confidence Threshold')
        plt.ylabel(key)
        plt.savefig(os.path.join(path,f'{key}.png'), 
                    dpi=300, 
                    bbox_inches='tight')