import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


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

