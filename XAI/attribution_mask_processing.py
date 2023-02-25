"""
attribution_mask_processing.py

Author: Leonardo Antunes Ferreira
Date:20/02/2023

To enhance the visualization of each attribution mask, we propose post-processing
steps to them, using computer vision techniques, enabling a more direct comparison
with the heatmap of the areas observed by humans, filtering out the less relevant
pixels of the input image.
"""
import cv2
import numpy as np
from typing import Tuple
from sklearn.cluster import KMeans


def attribution_mask_processing(attribution_mask: np.ndarray, 
                                n_clusters: int=5, 
                                ksize: int=11, 
                                sigma: int=0, 
                                alpha_thres: float=0.5) -> Tuple[np.ndarray, 
                                                                 np.ndarray]:
    """
    Enhance the visualisation of an attribution mask.
    
    inputs:
        attribution_mask: the attribution mask produced by an XAI method
        n_clusters: number of clusters to be considered on k-means
        ksize: kernel size for the Gaussian Blur
        sigma: kernel standard deviation for the Gaussian Blur
        alpha_thres: the standard deviation threshold used for defining if the
        pixel should be considered in the alpha channel
    
    returns:
        res2: the attribution_mask post processed
        alpha_channel: the alpha channel that can be applied to the image for 
        masking out the pixels
    """
    # Checks if the attribution_mask has 3 dimensions
    if len(attribution_mask.shape) <= 2:
        attribution_mask = np.expand_dims(attribution_mask, axis=-1)
    # Fit he kmeans to the flattened attribution_mask
    kmeans = KMeans(n_clusters=n_clusters, random_state=123)
    kmeans.fit(attribution_mask.reshape(-1, 1))
    # Assign to each pixel the corresponding cluster center from kmeans
    cluster_centers = kmeans.cluster_centers_
    result = cluster_centers[kmeans.labels_.flatten()]
    # Reshape to the original size
    result = result.reshape((attribution_mask.shape))
    result = np.squeeze(result)
    # Apply Gaussian Blur
    result = cv2.GaussianBlur(result,(ksize,ksize), sigma)
    # Creates the alpha channel
    alpha_channel = np.ones(np.squeeze(result).shape, dtype=result.dtype)
    alpha_channel[np.where(result <= result.mean() + 
                           result.std()*alpha_thres)[:2]] = 0

    return result, alpha_channel

