"""
face_detection.py

Author: Leonardo Antunes Ferreira
Date: 21/12/2022

This code is responsible for extracting the face coordinates and landmarks using
RetinaFace model from the Insightface package. The faces are cropped and saved
in a new folder. The facial coordinates are also saved on the .csv with all the
images names and data.
"""

import os
import pickle
from shutil import rmtree
import logging
import numpy as np

import cv2
import pandas as pd
from insightface.app import FaceAnalysis
from utils.utils import scale_coords, resize_landmarks

# Configure native logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Define dataset paths
dataset_path = os.path.join('Datasets', 'NewDataset', 'Images')
dataset_faces_path = os.path.join('Datasets', 'DatasetFaces', 'Images')
dataset_landmarks_path = os.path.join('Datasets', 'DatasetFaces', 'Landmarks')

# Read the CSV file
dataframe = pd.read_csv('iCOPE+UNIFESP_data.csv')

# Remove folders if present
try:
    rmtree(dataset_faces_path)
    logger.info(f"Removed existing directory: {dataset_faces_path}")
except Exception as e:
    logger.warning(f"Could not remove {dataset_faces_path}: {e}")

try:
    rmtree(dataset_landmarks_path)
    logger.info(f"Removed existing directory: {dataset_landmarks_path}")
except Exception as e:
    logger.warning(f"Could not remove {dataset_landmarks_path}: {e}")

# Create required directories
os.makedirs(dataset_faces_path, exist_ok=True)
os.makedirs(dataset_landmarks_path, exist_ok=True)
logger.info("Created directories for dataset faces and landmarks.")

# Instantiate RetinaFace model
logger.info("Initializing RetinaFace model...")
retinaface = FaceAnalysis(
    allowed_modules=['detection', 'landmark_2d_106'],
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
retinaface.prepare(ctx_id=0, det_size=(640, 640))
logger.info("RetinaFace model initialized.")

face_coordinates = []
keypoints_coordinates = []

logger.info("Starting face detection for images.")
for index, row in dataframe.iterrows():
    file_name = row['new_file_name']
    img_path = os.path.join(dataset_path, file_name)
    img = cv2.imread(img_path)
    
    if img is None:
        logger.warning(f"Image {file_name} not found or cannot be read.")
        face_coordinates.append([])
        keypoints_coordinates.append([])
        continue

    try:
        faces = retinaface.get(img)
        if len(faces) == 0:
            raise IndexError("No face detected.")
        face = faces[0]

        # Convert to int and fix any negative bbox values
        bbox = face['bbox'].astype('int')
        bbox[bbox < 0] = 0
        keypoints = face['kps'].astype('int')
        landmarks = face['landmark_2d_106'].astype('int')
        # Scale the landmarks based on the previous bbox, so it matches the facial image shape
        scaled_landmarks = [scale_coords(x, y, bbox) for x, y in landmarks]
        
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        cropped_face = img[y1:y2, x1:x2]

        # Resize the landmarks to match the new image size
        resized_landmarks = resize_landmarks(np.array(scaled_landmarks), cropped_face.shape[:2], (512, 512))

        cv2.imwrite(os.path.join(dataset_faces_path, file_name), cropped_face)
        face_coordinates.append(bbox.tolist())
        keypoints_coordinates.append(keypoints.tolist())

        landmarks_file = os.path.join(dataset_landmarks_path, f'{file_name.split(".")[0]}.pkl')
        with open(landmarks_file, 'wb') as f:
            pickle.dump(resized_landmarks, f)

    except IndexError:
        logger.warning(f"No faces were detected on image {file_name}")
        face_coordinates.append([])
        keypoints_coordinates.append([])

logger.info("Completed face detection for all images.")

# Update the dataframe with new columns and save to CSV
dataframe['face_coordinates'] = face_coordinates
dataframe['keypoints_coordinates'] = keypoints_coordinates
dataframe.to_csv('iCOPE+UNIFESP_data.csv', index=False)
logger.info("Updated CSV with face coordinates and keypoints saved.")
