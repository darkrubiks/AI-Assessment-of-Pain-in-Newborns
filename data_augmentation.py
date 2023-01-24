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
import cv2
import albumentations as A
import pandas as pd
from tqdm import tqdm
from ast import literal_eval
from utils import *


# Augmentation Pipeline, Affine transformation is always applied, the Horizontal
# Flip and RandomBrightnessContrast are applied with a 50% chance
transform = A.Compose([

        A.Affine(scale=(0.70, 1.5),
                 translate_percent=(-0.2,0.2),
                 rotate=(-30, 30),
                 shear=(-10, 10),
                 mode=cv2.BORDER_REPLICATE,
                 p=1),

        A.RandomBrightnessContrast(brightness_limit=0.2,
                                   contrast_limit=0.2,
                                   p=0.5),

        A.HorizontalFlip(p=0.5),
    ],

    keypoint_params=A.KeypointParams(format='xy'))

# Read the data from the .csv
iCOPE_UNIFESP_data = pd.read_csv('iCOPE+UNIFESP_data.csv')
iCOPE_UNIFESP_data['face_coordinates'] = iCOPE_UNIFESP_data['face_coordinates'].apply(lambda x: literal_eval(x))
iCOPE_UNIFESP_data['landmarks_coordinates'] = iCOPE_UNIFESP_data['landmarks_coordinates'].apply(lambda x: literal_eval(x))

folds_path = 'Datasets\\Folds'

# For each Fold the images are augmented 20 times, verifying that the landmarks
# are still in bounds of the new image
for fold in range(10):
    print(f'Augmenting Fold: {fold:02}')

    fold = str(fold)

    train_fold_path = os.path.join(folds_path, fold, 'Train')
    os.mkdir(os.path.join(train_fold_path, 'Landmarks'))

    for file_name in tqdm(os.listdir(train_fold_path)):
        if '.jpg' in file_name:
            # Read image
            img = cv2.imread(os.path.join(train_fold_path, file_name))
            # Get correspondent Face and Landmarks coordinates
            face_coords = iCOPE_UNIFESP_data[iCOPE_UNIFESP_data['new_file_name']==file_name]['face_coordinates'].values[0]
            landmarks_coords = iCOPE_UNIFESP_data[iCOPE_UNIFESP_data['new_file_name']==file_name]['landmarks_coordinates'].values[0]
            # Scale the landmarks to the cropped face
            scaled_landmarks = [scale_coords(x, y, face_coords[0], face_coords[1]) for x, y in landmarks_coords]# Save landmarks
            landmarks_file = open(os.path.join(train_fold_path, 'Landmarks', file_name.split('.jpg')[0]), 'wb')
            pickle.dump(scaled_landmarks, landmarks_file)
            landmarks_file.close()

            for i in range(20):

                transformed = transform(image=img, keypoints=scaled_landmarks)
                # Keep generating images until all landmarks are present
                while len(transformed['keypoints']) < 5:
                    transformed = transform(image=img, keypoints=scaled_landmarks)

                # Save image
                aug_file_name = f'{i:02}_AUG_{file_name}'
                cv2.imwrite(os.path.join(train_fold_path, aug_file_name), transformed['image'])
                # Save landmarks
                aug_landmarks_file = open(os.path.join(train_fold_path, 'Landmarks', aug_file_name.split('.jpg')[0]), 'wb')
                pickle.dump(transformed['keypoints'], aug_landmarks_file)
                aug_landmarks_file.close()





    