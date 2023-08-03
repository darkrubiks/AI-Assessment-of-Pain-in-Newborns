"""
leave_some_subject_out.py

Author: Gabriel de Almeida Sá Coutrin and Leonardo Antunes Ferreira
Date: 19/12/2022

This code is responsible for creating the folds for training and testing. The
folds are created based on the leave some subject out method developed during
Coutrin's Masters Degree, ensuring the same number of UNIFESP and COPE images
are used during training and testing, and also preventing subject leakage from
training to testing set. Only pain and no pain classifications are used. Also,
it creates a held-out-calibration set for further evaluation.
"""
import os
from shutil import copyfile, rmtree

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm

# Constants
FOLDS_FOLDER_PATH = os.path.join('Datasets','Folds')
DATASETS_FOLDER_PATH = os.path.join('Datasets','DatasetFaces')
CALIBRATION_FOLDER_PATH = os.path.join('Datasets','Calibration')
CALIBRATION_SIZE = 0.1
N_FOLDS = 10

# Remove folder if present
if os.path.exists(FOLDS_FOLDER_PATH):
    rmtree(FOLDS_FOLDER_PATH)
os.mkdir(FOLDS_FOLDER_PATH)

if os.path.exists(CALIBRATION_FOLDER_PATH):
    rmtree(CALIBRATION_FOLDER_PATH)
os.mkdir(CALIBRATION_FOLDER_PATH)

# Read the data
dataframe = pd.read_csv('iCOPE+UNIFESP_data.csv')

# Only keep pain and no pain labels
dataframe = dataframe[(dataframe['class']=='pain') |
                      (dataframe['class']=='no_pain')]

# Split the data into calibration and training/testing
_, X_calib, _, y_calib = train_test_split(
    dataframe["new_file_name"], 
    dataframe["class"], 
    test_size=CALIBRATION_SIZE, 
    random_state=42, 
    stratify=dataframe["class"]
)

# Copy the calibration images to a separate folder
for file_name in X_calib:
    src = os.path.join(DATASETS_FOLDER_PATH, 'Images', file_name)
    dst = os.path.join(CALIBRATION_FOLDER_PATH, file_name)
    copyfile(src, dst)

# Drop the calibration images from the training/testing data
dataframe = dataframe.drop(y_calib.index)

# Gather unique subjects
dataframe_no_duplicates = dataframe.drop_duplicates(subset=['new_subject'])

subjects = dataframe_no_duplicates['new_subject'].values
datasets = dataframe_no_duplicates['dataset'].values

train_subjects = []
test_subjects = []

# The StratifiedKFold is used to split the data between subjects but also consi-
# dering their original datasets, achieving balance between datasets
skf = StratifiedKFold(n_splits=N_FOLDS)

for train_index, test_index in skf.split(subjects, datasets):
    train_subjects.append(subjects[train_index])
    test_subjects.append(subjects[test_index])

for fold in range(N_FOLDS):
    # Create the fold folder
    fold_path = os.path.join(FOLDS_FOLDER_PATH, str(fold))
    os.mkdir(fold_path)

    # Create the train and test folders
    train_path = os.path.join(fold_path, "Train")
    os.mkdir(train_path)
    test_path = os.path.join(fold_path, "Test")
    os.mkdir(test_path)

    print('Copying Train Subjects')
    for train_subject in tqdm(train_subjects[int(fold)]):
        file_names = dataframe[dataframe['new_subject']==train_subject]['new_file_name']
        
        for file_name in file_names:
            src = os.path.join(DATASETS_FOLDER_PATH, 'Images', file_name)
            dst = os.path.join(train_path, file_name)
            copyfile(src, dst)

    print('Copying Test Subjects')
    for test_subject in tqdm(test_subjects[int(fold)]):
        file_names = dataframe[dataframe['new_subject']==test_subject]['new_file_name']
        
        for file_name in file_names:
            src = os.path.join(DATASETS_FOLDER_PATH, 'Images', file_name)
            dst = os.path.join(test_path, file_name)
            copyfile(src, dst)