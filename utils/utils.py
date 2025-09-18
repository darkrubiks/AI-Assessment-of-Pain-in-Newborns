import csv
import os
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml


def vis_keypoints(image: np.ndarray, 
                  keypoints: List[np.ndarray], 
                  color: np.ndarray=[0,255,0], 
                  diameter: int=15) -> None:
    """
    Plot keypoints on an image.
    """
    image = image.copy()

    for (x, y) in keypoints:
        cv2.circle(image, (int(x), int(y)), diameter, color, -1)
        
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.imshow(image)
    plt.show()


def scale_coords(x: int, 
                 y: int, 
                 face_bbox: List[int]) -> Tuple[int, int]:
    """
    Scale coordinates based on a new origin.
    """
    x1 = face_bbox[0]
    y1 = face_bbox[1]
    x2 = face_bbox[2]
    y2 = face_bbox[3]

    x = x - x1
    y = y - y1

    x = min(max(x, 0), x2-x1-5)
    y = min(max(y, 0), y2-y1-5)

    return int(x), int(y)


def resize_landmarks(landmarks: np.ndarray, 
                     original_shape: tuple, 
                     new_shape: tuple = (512, 512)) -> np.ndarray:
    """
    Resizes landmark coordinates from an original image coordinate system 
    to a new coordinate system with shape new_shape.
    """
    orig_h, orig_w = original_shape
    new_h, new_w = new_shape

    # Calculate scale factors for x (width) and y (height)
    scale_x = new_w / orig_w
    scale_y = new_h / orig_h

    # Apply the scaling to each coordinate.
    resized_landmarks = np.empty_like(landmarks, dtype=np.float32)
    resized_landmarks[:, 0] = landmarks[:, 0] * scale_x
    resized_landmarks[:, 1] = landmarks[:, 1] * scale_y

    return resized_landmarks


def load_config(config_file: str) -> dict:
    """
    Loads a .yaml configuration file.
    """
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
        
    return config


def write_to_csv(filename, **kwargs):
    """
    Writes information to .csv file. If file already exists data will
    be appended.
    """
    mode = 'a' if os.path.exists(filename) else 'w'
 
    with open(filename, mode, newline='') as file:
        writer = csv.writer(file)
        if mode == 'w':
            writer.writerow(kwargs.keys())  # Write header row if the file is new
        writer.writerow(kwargs.values())


def create_folder(path):
    """
    Tries to create a folder on the informed path.
    """
    try:
        os.makedirs(path)
    except FileExistsError:
        pass
