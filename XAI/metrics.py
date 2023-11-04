"""
metrics.py

Author: Leonardo Antunes Ferreira
Date: 04/11/2023

XAI metrics.
"""
import numpy as np


def calculate_xai_score(xai_mask, region_masks):
    """
    This function is responsible for calculating a XAI score which 
    will return the percentage of important pixels in each facial 
    region. An important pixel is defined base on its value if it's 
    close to 1 more relvant this pixels is, and close to 0 less 
    important. This fucntion will retunr a dictionary containing 
    the normalized inportance values for each given facial region.

    Parameters
    ----------
    xai_mask : the XAI atribution mask in range [0 - 1]

    region_masks : a dictionary containing the key value as the 
    region name and the value as the image mask in range [0 - 1]

    Returns
    -------
    normalized_scores : a dictionary containing the given region 
    names with its respective normalized score summing up to 1
    """
    region_scores = {}
    # For each region calculates the average pixel value, regions 
    # with more importace will have larger values
    for region, region_mask in region_masks.items():
        region_importance = np.mean(xai_mask * region_mask)
        region_scores[region] = region_importance

    total_score = sum(region_scores.values())

    # Normalize the scores so it sums up to 1
    normalized_scores = {}
    for region, score in region_scores.items():
        normalized_score = score / total_score
        normalized_scores[region] = normalized_score

    return normalized_scores
