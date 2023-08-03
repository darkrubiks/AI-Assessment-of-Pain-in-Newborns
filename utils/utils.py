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
                 x_new_origin: int, 
                 y_new_origin: int) -> Tuple[int, int]:
    """
    Scale coordinates based on a new origin.
    """
    scaled_x = x - x_new_origin
    scaled_y = y - y_new_origin

    return scaled_x, scaled_y


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
        os.mkdir(path)
    except FileExistsError:
        pass
