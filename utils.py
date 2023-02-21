import cv2
import matplotlib.pyplot as plt


def vis_keypoints(image, keypoints, color=(0,255,0), diameter=15):
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

def scale_coords(x, y, x_new_origin, y_new_origin):
    """
    Scale coordinates based on a new origin.
    """
    scaled_x = x - x_new_origin
    scaled_y = y - y_new_origin

    return scaled_x, scaled_y

