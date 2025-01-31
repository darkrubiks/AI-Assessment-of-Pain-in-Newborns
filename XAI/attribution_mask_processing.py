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
from sklearn.cluster import KMeans, MiniBatchKMeans  # Use MiniBatchKMeans for speed

def attribution_mask_processing(attribution_mask: np.ndarray, 
                                n_clusters: int = 5, 
                                ksize: int = 11, 
                                sigma: int = 0, 
                                alpha_thres: float = 0.5,
                                use_mini_batch: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Enhance the visualization of an attribution mask with clustering, blurring,
    and thresholding – optimized for speed.
    
    Parameters
    ----------
    attribution_mask : np.ndarray
        The attribution mask produced by an XAI method.
    n_clusters : int, optional
        Number of clusters for KMeans clustering, by default 5.
    ksize : int, optional
        Kernel size for the Gaussian Blur, by default 11.
    sigma : int, optional
        Kernel standard deviation for the Gaussian Blur, by default 0.
    alpha_thres : float, optional
        Threshold factor for defining the alpha channel, by default 0.5.
    use_mini_batch : bool, optional
        If True, uses MiniBatchKMeans for speed, by default True.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - Post-processed attribution mask normalized to [0,1].
        - Alpha channel mask with pixels above threshold set to 1, others 0.
    """
    # Ensure the mask is at least 3D for clustering (H, W, 1)
    if attribution_mask.ndim <= 2:
        attribution_mask = attribution_mask[..., np.newaxis]
    
    # Flatten the mask for clustering.
    flat_mask = attribution_mask.reshape(-1, 1)
    
    # Choose clustering method.
    if use_mini_batch:
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=123, batch_size=1024)
    else:
        kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=123)
    
    # Fit clustering and assign each pixel to its cluster center.
    kmeans.fit(flat_mask)
    clustered = kmeans.cluster_centers_[kmeans.labels_].reshape(attribution_mask.shape)
    
    # Remove extra singleton dimensions (if any).
    result = np.squeeze(clustered)
    
    # Apply Gaussian blur.
    result = cv2.GaussianBlur(result, (ksize, ksize), sigma)
    
    # Normalize to the range [0, 1] (with epsilon to prevent div-by-zero).
    r_min, r_max = result.min(), result.max()
    result = (result - r_min) / (r_max - r_min + 1e-8)
    
    # Compute threshold and build alpha channel in one vectorized operation.
    thresh = result.mean() + result.std() * alpha_thres
    alpha_channel = (result > thresh).astype(result.dtype)
    
    return result, alpha_channel
