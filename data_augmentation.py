"""
data_augmentation.py

Author: Leonardo Antunes Ferreira
Date: 03/01/2023

This code is responsible for augmenting the train set. The Data Augmentation
pipeline will generate 20 new images from 1 single face image. The facial land-
marks are also augmented.
"""
import os
import pickle
from ast import literal_eval

import albumentations as A
import cv2
import pandas as pd
from tqdm import tqdm

from utils.utils import create_folder, scale_coords

# Constants
FOLDS_FOLDER_PATH = os.path.join('Datasets', 'Folds')
CALIBRATION_FOLDER_PATH = os.path.join('Datasets', 'Calibration')
N_FOLDS = len(os.listdir(FOLDS_FOLDER_PATH))
AUGMENTED_IMAGES = 20
AUGMENTED_SUFFIX = "_AUG_"


def resize_original_img(path, file_name):
    # Read image
    img = cv2.imread(os.path.join(path, file_name))

    # Get correspondent Face and keypoints coordinates
    face_coords = iCOPE_UNIFESP_data[iCOPE_UNIFESP_data['new_file_name']==file_name]['face_coordinates'].values[0]
    keypoints_coords = iCOPE_UNIFESP_data[iCOPE_UNIFESP_data['new_file_name']==file_name]['keypoints_coordinates'].values[0]

    # Scale the keypoints to the cropped face
    scaled_keypoints = [scale_coords(x, y, face_coords) for x, y in keypoints_coords]
    resized = resize(image=img, keypoints=scaled_keypoints)

    # Save keypoints and resized image
    cv2.imwrite(os.path.join(path, file_name), resized['image'])

    with open(os.path.join(path, 'Keypoints', file_name.split('.jpg')[0] + ".pkl"), 'wb') as f:
        pickle.dump(resized['keypoints'], f)

    return img, scaled_landmarks

# Augmentation Pipeline, Affine transformation is always applied, the Horizontal
# Flip and RandomBrightnessContrast are applied with a 50% chance. All images and
# and keypoints are resized to 512x512 
transform = A.Compose(
    [
        A.Affine(
            scale=(0.70, 1.5),
            translate_percent=(-0.2, 0.2),
            rotate=(-30, 30),
            shear=(-10, 10),
            mode=cv2.BORDER_REPLICATE,
            p=1.0,
        ),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HorizontalFlip(p=0.5),
    ],
    keypoint_params=A.KeypointParams(format="xy"),
)

resize = A.Compose(
    [
        A.Resize(height=556, width=556, interpolation=cv2.INTER_CUBIC, p=1.0),
        A.CenterCrop(height=512, width=512, p=1.0),
    ],
    keypoint_params=A.KeypointParams(format="xy"),
)

# Read the data from the .csv
iCOPE_UNIFESP_data = pd.read_csv('iCOPE+UNIFESP_data.csv')
iCOPE_UNIFESP_data['face_coordinates'] = iCOPE_UNIFESP_data['face_coordinates'].apply(lambda x: literal_eval(x))
iCOPE_UNIFESP_data['keypoints_coordinates'] = iCOPE_UNIFESP_data['keypoints_coordinates'].apply(lambda x: literal_eval(x))

# Apply to Calibration set only resizing
create_folder(os.path.join(CALIBRATION_FOLDER_PATH, 'Keypoints'))
print('Applying to Calibration Set')
for file_name in tqdm(os.listdir(CALIBRATION_FOLDER_PATH)):
    if file_name.endswith('.jpg'):
        _ = resize_original_img(CALIBRATION_FOLDER_PATH, file_name)

# For each Fold the images are augmented 20 times, verifying that the Keypoints
# are still in bounds of the new image
for fold in range(N_FOLDS):
    print(f'\nAugmenting Fold: {fold:02}')

    fold = str(fold)

    train_fold_path = os.path.join(FOLDS_FOLDER_PATH , fold, 'Train')
    test_fold_path = os.path.join(FOLDS_FOLDER_PATH , fold, 'Test')
    create_folder(os.path.join(train_fold_path, 'Keypoints'))
    create_folder(os.path.join(test_fold_path, 'Keypoints'))

    print('Applying to Test Set')
    for file_name in tqdm(os.listdir(test_fold_path)):
        if file_name.endswith('.jpg'):
            _ = resize_original_img(test_fold_path, file_name)

    print('Applying to Train Set')
    for file_name in tqdm(os.listdir(train_fold_path)):
        if file_name.endswith('.jpg'):
            img, scaled_landmarks = resize_original_img(train_fold_path, file_name)

            for i in range(AUGMENTED_IMAGES):
                transformed = transform(image=img, keypoints=scaled_landmarks)

                # Keep generating images until all Keypoints are present
                while len(transformed['keypoints']) < 5:
                    transformed = transform(image=img, keypoints=scaled_landmarks)

                # Save image
                aug_file_name = f'{i:02}{AUGMENTED_SUFFIX}{file_name}'
                cv2.imwrite(os.path.join(train_fold_path, aug_file_name), transformed['image'])

                # Save Keypoints
                aug_landmarks_file = os.path.join(train_fold_path, 'Keypoints', aug_file_name.split('.jpg')[0] + ".pkl")
                with open(aug_landmarks_file, 'wb') as f:
                    pickle.dump(transformed['keypoints'], f)