"""
face_detection.py.py

Author: Leonardo Antunes Ferreira
Date: 21/12/2022

This code is responsible for extracting the face coordinates and landmarks using
RetinaFace model from the Insgightface package. The faces are cropped and saved
in a new folder. The facial coordinates are also saved on the .csv with all the
images names and data.
"""
import cv2
import os
import insightface
import pandas as pd
from shutil import rmtree
from tqdm import tqdm


dataset_path = 'Datasets\\NewDataset\\Images\\'
dataset_faces_path = 'Datasets\\DatasetFaces\\Images\\'

dataframe = pd.read_csv('iCOPE+UNIFESP_data.csv')

# Remove folder if present
try:
    rmtree(dataset_faces_path)
except:
    pass

# Create main folder
os.makedirs(dataset_faces_path)

# Instantiate RetinaFace model
retinaface = insightface.model_zoo.get_model('retinaface_r50_v1')
retinaface.prepare(ctx_id=-1) 

face_coordinates = []
landmark_coordinates = []

for _, row in tqdm(dataframe.iterrows()):

    file_name = row['new_file_name']

    img = cv2.imread(os.path.join(dataset_path, file_name))

    bbox, landmarks = retinaface.detect(img, scale=0.5)

    bbox = bbox[0]
    landmarks = landmarks[0]

    x = abs(int(bbox[0]))
    y = abs(int(bbox[1]))
    w = abs(int(bbox[2])) - x
    h = abs(int(bbox[3])) - y

    face = img[y:y+h,x:x+w]

    cv2.imwrite(os.path.join(dataset_faces_path, file_name), face)

    face_coordinates.append(bbox)
    landmark_coordinates.append(landmarks)

# Assign the new columns to the dataframe and save the .csv
dataframe['face_coordinates'] = face_coordinates
dataframe['landmarks_coordinates'] = landmark_coordinates

dataframe.to_csv('iCOPE+UNIFESP_data.csv', index=False)

