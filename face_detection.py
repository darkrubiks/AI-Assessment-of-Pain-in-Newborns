"""
face_detection.py

Author: Leonardo Antunes Ferreira
Date: 21/12/2022

This code is responsible for extracting the face coordinates and landmarks using
RetinaFace model from the Insgightface package. The faces are cropped and saved
in a new folder. The facial coordinates are also saved on the .csv with all the
images names and data.
"""
import os
from shutil import rmtree

import cv2
import insightface
import numpy as np
import pandas as pd
from insightface.app import FaceAnalysis
from tqdm import tqdm

dataset_path = os.path.join('Datasets','NewDataset','Images')
dataset_faces_path = os.path.join('Datasets','DatasetFaces','Images')

dataframe = pd.read_csv('iCOPE+UNIFESP_data.csv')

# Remove folder if present
try:
    rmtree(dataset_faces_path)
except:
    pass

# Create main folder
os.makedirs(dataset_faces_path)

# Instantiate RetinaFace model
retinaface = FaceAnalysis(allowed_modules=['detection'], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
retinaface.prepare(ctx_id=0, det_size=(640, 640)) 

face_coordinates = []
landmark_coordinates = []

for _, row in tqdm(dataframe.iterrows()):
    file_name = row['new_file_name']

    img = cv2.imread(os.path.join(dataset_path, file_name))

    try:
        faces = retinaface.get(img)[0]

        # Convert to int, and remove the detection accuracy from bbox
        bbox = faces['bbox'].astype('int')
        bbox[bbox < 0] = 0
        landmarks = faces['kps'].astype('int')

        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]

        face = img[y1:y2,x1:x2]

        cv2.imwrite(os.path.join(dataset_faces_path, file_name), face)

        face_coordinates.append(bbox.tolist())
        landmark_coordinates.append(landmarks.tolist())

    except IndexError:
        print(f"No faces were detected on image {file_name}")
        face_coordinates.append([])
        landmark_coordinates.append([])

# Assign the new columns to the dataframe and save the .csv
dataframe['face_coordinates'] = face_coordinates
dataframe['landmarks_coordinates'] = landmark_coordinates

dataframe.to_csv('iCOPE+UNIFESP_data.csv', index=False)