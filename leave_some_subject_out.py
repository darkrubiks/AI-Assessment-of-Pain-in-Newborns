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
import os
import pandas as pd
from shutil import copyfile, rmtree
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm


dataframe = pd.read_csv('iCOPE+UNIFESP_data.csv')

# Secting only the pain and no pain labels
dataframe = dataframe[(dataframe['class']=='pain') |
                      (dataframe['class']=='no_pain')]

dataframe_no_duplicates = dataframe.drop_duplicates(subset=['new_subject'])

subjects = dataframe_no_duplicates['new_subject'].values
datasets = dataframe_no_duplicates['dataset'].values

train_subjects = []
test_subjects = []

train_datasets = []
test_datasets = []

# The StratifiedKFold is used to split the data between subjects but also consi-
# dering their original datasets, achieving balance between datasets
n_folds = 10
skf = StratifiedKFold(n_splits=n_folds)

for train_index, test_index in skf.split(subjects, datasets):
    
    train_subjects.append(subjects[train_index])
    test_subjects.append(subjects[test_index])
    train_datasets.append(datasets[train_index])
    test_datasets.append(datasets[test_index])

# Copy the images for each fold into a new folder
folds_path = 'Datasets\\Folds'
dataset_path = 'Datasets\\NewDataset'

# Remove folder if present
try:
    rmtree(folds_path)
except:
    pass

# Create main folder
os.mkdir(folds_path)

# For each fold create a folder containing Train and Test Images and Heatmaps
for fold in range(n_folds):

    print(f'Creating Folder for fold {fold}')
    fold = str(fold)

    os.mkdir(os.path.join(folds_path, fold))

    os.mkdir(os.path.join(folds_path, fold, 'Train'))

    os.mkdir(os.path.join(folds_path, fold, 'Test'))

    print('Copying Train Subjects')
    for train_subject in tqdm(train_subjects[int(fold)]):

        file_names = dataframe[dataframe['new_subject']==train_subject]['new_file_name']
        
        for file_name in file_names:

            src = os.path.join(dataset_path, 'Images', file_name)
            dst = os.path.join(folds_path, fold, 'Train', file_name)

            copyfile(src, dst)

    print('Copying Test Subjects')
    for test_subject in tqdm(test_subjects[int(fold)]):

        file_names = dataframe[dataframe['new_subject']==test_subject]['new_file_name']
        
        for file_name in file_names:

            src = os.path.join(dataset_path, 'Images', file_name)
            dst = os.path.join(folds_path, fold, 'Test', file_name)

            copyfile(src, dst)


   


