import os
from shutil import copyfile, rmtree

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from utils.utils import create_folder

# Constants
FOLDS_FOLDER_PATH = os.path.join('Datasets', 'Folds')
DATASETS_FOLDER_PATH = os.path.join('Datasets', 'DatasetFaces')
CALIBRATION_FOLDER_PATH = os.path.join('Datasets', 'Calibration')
CALIBRATION_SIZE = 0.1
N_FOLDS = 5

def copy_files(src_files, dst_folder):
    """
    Tries to create a folder on the informed path.
    """
    for src_file in src_files:
        src = os.path.join(DATASETS_FOLDER_PATH, 'Images', src_file)
        dst = os.path.join(dst_folder, src_file)
        copyfile(src, dst)

if os.path.exists(FOLDS_FOLDER_PATH):
    rmtree(FOLDS_FOLDER_PATH)
create_folder(FOLDS_FOLDER_PATH)

if os.path.exists(CALIBRATION_FOLDER_PATH):
    rmtree(CALIBRATION_FOLDER_PATH)
create_folder(CALIBRATION_FOLDER_PATH)

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
copy_files(X_calib, CALIBRATION_FOLDER_PATH)

# Drop the calibration images from the training/testing data
dataframe = dataframe.drop(y_calib.index)

# Gather unique subjects
unique_subjects = list(set(dataframe['new_subject']))
datasets = dataframe.drop_duplicates(subset=['new_subject'])['dataset'].values

train_subjects = []
test_subjects = []

# The StratifiedKFold is used to split the data between subjects but also consider their original datasets, achieving balance between datasets
skf = StratifiedKFold(n_splits=N_FOLDS)

for train_index, test_index in skf.split(unique_subjects, datasets):
    train_subjects.append([unique_subjects[i] for i in train_index])
    test_subjects.append([unique_subjects[i] for i in test_index])

for fold in range(N_FOLDS):
    # Create the fold folder
    fold_path = os.path.join(FOLDS_FOLDER_PATH, str(fold))
    create_folder(fold_path)

    # Create the train and test folders
    train_path = os.path.join(fold_path, "Train")
    create_folder(train_path)
    test_path = os.path.join(fold_path, "Test")
    create_folder(test_path)

    # Copy Train Subjects
    copy_files(dataframe[dataframe['new_subject'].isin(train_subjects[fold])]['new_file_name'], train_path)

    # Copy Test Subjects
    copy_files(dataframe[dataframe['new_subject'].isin(test_subjects[fold])]['new_file_name'], test_path)
