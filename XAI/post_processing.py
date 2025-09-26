"""
post_processing.py

Author: Leonardo Antunes Ferreira
Date: 20/02/2023

Post-processing utilities for attribution masks to improve visualization and
highlight the most relevant image regions using computer vision techniques.
"""
import cv2
import numpy as np
from typing import Tuple
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.signal import fftconvolve

_EPSILON = 1e-8


def kmeans_post_processing(attribution_mask: np.ndarray,
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
        If True, uses MiniBatchKMeans for speed, by default False.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - Post-processed attribution mask normalized to [0, 1].
        - Alpha channel mask with pixels above threshold set to 1, others 0.
    """
    if attribution_mask.ndim <= 2:
        attribution_mask = attribution_mask[..., np.newaxis]

    flat_mask = attribution_mask.reshape(-1, 1)

    if use_mini_batch:
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=123, batch_size=1024)
    else:
        kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=123)

    kmeans.fit(flat_mask)
    clustered = kmeans.cluster_centers_[kmeans.labels_].reshape(attribution_mask.shape)

    result = np.squeeze(clustered)
    result = cv2.GaussianBlur(result, (ksize, ksize), sigma)

    r_min, r_max = result.min(), result.max()
    result = (result - r_min) / (r_max - r_min + _EPSILON)

    thresh = result.mean() + result.std() * alpha_thres
    alpha_channel = (result > thresh).astype(np.uint8)

    return result * alpha_channel


def tobii_cspline_kernel(radius_px: int) -> np.ndarray:
    """
    Create a Tobii C-spline kernel with mass-preserving normalization.

    Parameters
    ----------
    radius_px : int
        Radius of the kernel in pixels.

    Returns
    -------
    np.ndarray
        Normalized kernel ready for convolution.
    """
    radius = float(radius_px)
    y, x = np.ogrid[-radius_px: radius_px + 1, -radius_px: radius_px + 1]
    r = np.sqrt(x * x + y * y)

    u = np.clip(r / radius, 0.0, 1.0)
    smoothstep = u * u * (3.0 - 2.0 * u)
    weights = 1.0 - smoothstep
    weights[r > radius] = 0.0
    weights /= weights.sum()

    return weights


def tobii_post_processing(heatmap: np.ndarray,
                          radius_px: int = 50,
                          clip: bool = True) -> np.ndarray:
    """
    Smooth a heatmap with the Tobii C-spline kernel.

    Parameters
    ----------
    heatmap : np.ndarray
        2D float array representing the heatmap.
    radius_px : int, optional
        Radius of the smoothing kernel in pixels, by default 50.
    clip : bool, optional
        If True, clip the output to the [0, 1] range, by default True.

    Returns
    -------
    np.ndarray
        Smoothed heatmap.
    """
    kernel = tobii_cspline_kernel(radius_px)
    convolved = fftconvolve(heatmap.astype(np.float64), kernel, mode='same')

    if clip:
        minimum, maximum = convolved.min(), convolved.max()
        if maximum > minimum:
            convolved = (convolved - minimum) / (maximum - minimum)
        convolved = np.clip(convolved, 0.0, 1.0)

    return convolved


def get_top_k_pixels(heatmap: np.ndarray,
                     k_percent: float = 10.0,
                     binary: bool = True) -> np.ndarray:
    """
    Build a binary mask containing the top-k percent most important pixels.

    Parameters
    ----------
    heatmap : np.ndarray
        Importance scores for each pixel.
    k_percent : float, optional
        Percentage of pixels to keep, by default 10.0.

    Returns
    -------
    np.ndarray
        Binary mask with the same shape as `importance_map`.
    """
    flat_map = heatmap.flatten()
    threshold_value = np.percentile(flat_map, 100 - k_percent)
    alpha_tresh = heatmap >= threshold_value

    if binary:
        return alpha_tresh.astype(np.int32)
    else:
        return heatmap * alpha_tresh
