"""
leave_some_subject_out.py

Author: Gabriel de Almeida Sá Coutrin and Leonardo Antunes Ferreira
Date: 19/12/2022

This code is responsible for creating the folds for training and testing. The
folds are created based on the leave some sobject out method developed during
Coutrin's Masters Degree, ensuring the same number of UNIFESP and COPE images
are used during training and testing, and also preventing subject leakage from
training to testing set. Only pain and no pain classifications are used.
"""
import pandas as pd
from sklearn.model_selection import StratifiedKFold


dataframe = pd.read_csv('iCOPE+UNIFESP_data.csv')

# Secting only the pain and no pain labels
dataframe = dataframe[(dataframe['class']=='pain') |
                      (dataframe['class']=='no_pain')]

dataframe.drop_duplicates(subset=['new_subject'], inplace=True)

subjects = dataframe['new_subject'].values
datasets = dataframe['dataset'].values

train_subjects = []
test_subjects = []

train_datasets = []
test_datasets = []

# The StratifiedKFold is used to split the data between subjects but also consi-
# dering their original datasets, achieving balance between datasets
skf = StratifiedKFold(n_splits=10)

for train_index, test_index in skf.split(subjects, datasets):
    
    train_subjects.append(subjects[train_index])
    test_subjects.append(subjects[test_index])
    train_datasets.append(datasets[train_index])
    test_datasets.append(datasets[test_index])

