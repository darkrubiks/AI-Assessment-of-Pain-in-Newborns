import os
from shutil import copyfile, rmtree

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from utils.utils import create_folder

import logging

# Configure native logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Constants
FOLDS_FOLDER_PATH = os.path.join('Datasets', 'Folds')
DATASETS_FOLDER_PATH = os.path.join('Datasets', 'DatasetFaces')
N_FOLDS = 5

def copy_files(src_files, dst_folder):
    """
    Copy files from the dataset folder to the destination folder.
    """
    for src_file in src_files:
        src = os.path.join(DATASETS_FOLDER_PATH, 'Images', src_file)
        dst = os.path.join(dst_folder, src_file)
        copyfile(src, dst)
    logger.info(f"Copied {len(src_files)} files to {dst_folder}")

# Remove and create Folds folder
if os.path.exists(FOLDS_FOLDER_PATH):
    rmtree(FOLDS_FOLDER_PATH)
    logger.info(f"Removed existing folder: {FOLDS_FOLDER_PATH}")
create_folder(FOLDS_FOLDER_PATH)
logger.info(f"Created folder: {FOLDS_FOLDER_PATH}")

# Read the data
logger.info("Reading CSV file 'iCOPE+UNIFESP_data.csv'")
dataframe = pd.read_csv('iCOPE+UNIFESP_data.csv')

# Remove entries without face detection
dataframe = dataframe[dataframe['face_coordinates'] != "[]"]
logger.info(f"Filtered out entries without face detection. Remaining entries: {len(dataframe)}")

# Only keep pain and no pain labels
dataframe = dataframe[(dataframe['class'] == 'pain') | (dataframe['class'] == 'nopain')]
logger.info(f"Filtered dataframe to include only 'pain' and 'nopain' classes. Remaining entries: {len(dataframe)}")

# Only keep selected datasets
dataframe = dataframe[(dataframe['dataset'] == 'iCOPE') | (dataframe['dataset'] == 'UNIFESP')]
logger.info(f"Filtered dataframe to include only 'iCOPE' and 'UNIFESP' datasets. Remaining entries: {len(dataframe)}")

# Gather unique subjects and their corresponding datasets
unique_subjects = list(set(dataframe['new_subject']))
datasets = dataframe.drop_duplicates(subset=['new_subject'])['dataset'].values
logger.info(f"Found {len(unique_subjects)} unique subjects.")

train_subjects = []
test_subjects = []

# Use StratifiedKFold to split subjects while maintaining balance across datasets
skf = StratifiedKFold(n_splits=N_FOLDS)
for train_index, test_index in skf.split(unique_subjects, datasets):
    train_subjects.append([unique_subjects[i] for i in train_index])
    test_subjects.append([unique_subjects[i] for i in test_index])
logger.info("Completed splitting subjects into folds using StratifiedKFold.")

# Create folds and copy files accordingly
for fold in range(N_FOLDS):
    logger.info(f"Processing fold {fold}")
    # Create fold folder
    fold_path = os.path.join(FOLDS_FOLDER_PATH, str(fold))
    create_folder(fold_path)
    logger.info(f"Created fold folder: {fold_path}")

    # Create Train and Test folders
    train_path = os.path.join(fold_path, "Train")
    create_folder(train_path)
    logger.info(f"Created train folder: {train_path}")

    test_path = os.path.join(fold_path, "Test")
    create_folder(test_path)
    logger.info(f"Created test folder: {test_path}")

    # Copy training images
    train_files = dataframe[dataframe['new_subject'].isin(train_subjects[fold])]['new_file_name']
    copy_files(train_files, train_path)
    logger.info(f"Copied {len(train_files)} training images to {train_path}")

    # Copy testing images
    test_files = dataframe[dataframe['new_subject'].isin(test_subjects[fold])]['new_file_name']
    copy_files(test_files, test_path)
    logger.info(f"Copied {len(test_files)} testing images to {test_path}")

# --- Additional splitting of the dataset using train_test_split ---
# Global variable for train size (adjust this value as needed)
TRAIN_SIZE = 0.9  # Example: 90% training, 10% testing

# Split the dataset into train and test sets, stratifying based on the 'class' column
train_set, test_set = train_test_split(
    dataframe,
    train_size=TRAIN_SIZE,
    stratify=dataframe['class'],
    random_state=42
)
logger.info(f"Dataset split into train ({len(train_set)}) and test ({len(test_set)}) sets with train size: {TRAIN_SIZE}")

# --- Save the train_test_split data into TrainAll folder with subfolders Train and Test ---
TRAIN_ALL_FOLDER = os.path.join('Datasets', 'Folds', 'TrainAll')
TRAIN_ALL_TRAIN_FOLDER = os.path.join(TRAIN_ALL_FOLDER, 'Train')
TRAIN_ALL_TEST_FOLDER = os.path.join(TRAIN_ALL_FOLDER, 'Test')

# Remove existing TrainAll folder if it exists
if os.path.exists(TRAIN_ALL_FOLDER):
    rmtree(TRAIN_ALL_FOLDER)
    logger.info(f"Removed existing folder: {TRAIN_ALL_FOLDER}")

# Create TrainAll and its subfolders
create_folder(TRAIN_ALL_FOLDER)
create_folder(TRAIN_ALL_TRAIN_FOLDER)
logger.info(f"Created folder: {TRAIN_ALL_TRAIN_FOLDER}")
create_folder(TRAIN_ALL_TEST_FOLDER)
logger.info(f"Created folder: {TRAIN_ALL_TEST_FOLDER}")

# Copy training images to TrainAll/Train
train_all_train_files = train_set['new_file_name']
copy_files(train_all_train_files, TRAIN_ALL_TRAIN_FOLDER)
logger.info(f"Copied {len(train_all_train_files)} training images to {TRAIN_ALL_TRAIN_FOLDER}")

# Copy testing images to TrainAll/Test
train_all_test_files = test_set['new_file_name']
copy_files(train_all_test_files, TRAIN_ALL_TEST_FOLDER)
logger.info(f"Copied {len(train_all_test_files)} testing images to {TRAIN_ALL_TEST_FOLDER}")
