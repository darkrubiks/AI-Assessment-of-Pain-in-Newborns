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


def plot_calibration_curve(probs: np.ndarray, 
                           labels: np.ndarray, 
                           n_bins: int=10,
                           path: str=os.getcwd()) -> None:
    """
    Plots the calibration curve. It is also possible to include the amount
    of samples in each bin as a circle with dynamic radius.

    Parameters
    ----------
    probs : probability of the positive class

    labels : true labels as binary targets

    n_bins : number of bins to discretize

    plot_samples : if True it will also plot the amount of samples in each
    bin like a circle where a bigger radius means more samples

    path : where to save the plot image. Defaults to current directory.
    """
    fig = plt.figure()
    gs = fig.add_gridspec(2, 1,  height_ratios=(3, 1), left=0.1, right=0.9, bottom=0.1, top=0.9, hspace=0.05)

    ax_curve = fig.add_subplot(gs[0])
    ax_hist = fig.add_subplot(gs[1], sharex=ax_curve)
    
    prob_true, prob_pred, _ =  calibration_curve(probs, labels, n_bins)
    
    ece = ECE(probs, labels, n_bins)

    ax_curve.plot(prob_pred, prob_true,  linestyle='-', marker='o', color=COLOR, label=f'ECE = {ece:.4f}')
    ax_curve.plot(np.arange(0,1.1,0.1), np.arange(0,1.1,0.1), 'k--')
    ax_hist.hist(probs, bins=n_bins, color=COLOR, edgecolor='k')

    ax_curve.set_title('Calibration Curve')
    ax_curve.set_ylabel('Fraction of Positives')
    ax_curve.legend()

    ax_hist.set_xlabel('Mean Predicted Confidence')
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

    plt.plot(fpr, tpr, color=COLOR, label=f'AUC = {auc:.4f}')
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
    
    plt.plot(recall, precision, color=COLOR, label=f'AP = {ap:.4f}')
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
    by changing the confidence threshold.

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
        plt.plot(threshold[nonzero], above_threshold[nonzero], color=COLOR, label=f'{key} at 0.5 = {result:.4f}')
        plt.xlabel('Confidence Threshold')
        plt.ylabel(key)
        plt.legend()
        plt.savefig(os.path.join(path,f'{key}.png'), 
                    dpi=300, 
                    bbox_inches='tight')
        plt.close()