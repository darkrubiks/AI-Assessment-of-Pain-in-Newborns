"""
metrics.py

Author: Leonardo Antunes Ferreira
Date: 04/11/2023

XAI metrics.
"""
import numpy as np
import cv2


def calculate_xai_score(xai_mask: np.ndarray, region_masks: dict) -> dict:
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
    sorted_dict : a dictionary containing the given region 
    names with its respective normalized score summing up to 1 sorted
    by descending order of importance
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

    # Sort the dictionary by values in ascending order
    sorted_dict = dict(sorted(normalized_scores.items(), key=lambda item: item[1], reverse=True))

    return sorted_dict


def create_face_regions_masks(keypoints: np.ndarray) -> dict:
    """
    Creates regions masks based on 68 facial keypoints, for more
    information see https://github.com/nttstar/insightface-resources/blob/master/alignment/images/2d106markup.jpg
    or the publication https://doi.org/10.5753/wvc.2021.18914.

    Parameters
    ----------
    keypoints : the 68 keypoints extracted from the face

    Returns
    -------
    region_masks : a dictionary containing the binary region_masks 

    See also : P. Domingues, et al. "Neonatal Face Mosaic: An areas-of-interest 
    segmentation method based on 2D face images". Anais do XVII Workshop de Visão 
    Computacional, 2021. 

    doi: https://doi.org/10.5753/wvc.2021.18914.
    """
    right_eyebrown = keypoints[[43, 44, 45, 47,
                                46, 50, 51, 49, 48]].astype(np.int32)
    left_eyebrown = keypoints[[97, 98, 99, 100,
                               101, 105, 104, 103, 102]].astype(np.int32)

    right_eye = keypoints[[81, 89, 90, 87, 91, 93, 101, 100, 99, 98, 97]].astype(np.int32)
    left_eye = keypoints[[75, 39, 37, 33, 36, 35, 43, 44, 45, 47, 46]].astype(np.int32)

    nose = keypoints[[72, 75, 76, 77, 78, 79,
                      80, 85, 84, 83, 82, 81]].astype(np.int32)

    between_eyes = keypoints[[50, 46, 75, 72, 81, 97, 102]].astype(np.int32)

    mouth = keypoints[[52, 64, 63, 67, 68, 61,
                       58, 59, 53, 56, 55]].astype(np.int32)
    
    chin = keypoints[[5, 6, 7, 8, 0, 24, 23, 22, 21,
                      58, 59, 53, 56, 55]].astype(np.int32)

    right_nasolabial_fold = keypoints[[ 5, 4, 3, 77, 78, 79, 52]].astype(np.int32)
    left_nasolabial_fold = keypoints[[20, 21, 22, 61, 85, 84, 83]].astype(np.int32)

    right_cheek = keypoints[[30, 31, 32, 18, 19, 20, 83, 82]].astype(np.int32)
    left_cheek = keypoints[[14, 15, 16, 2, 3, 77, 76]].astype(np.int32)

    forehead_left = 2 * keypoints[1] - keypoints[[1, 9, 10, 11, 12, 13]]
    forehead_right = 2 * keypoints[17] - keypoints[[29, 28, 27, 26, 25, 17]]

    forehead = np.vstack((forehead_left.astype(np.int32),
                         forehead_right.astype(np.int32)))

    outside = keypoints[[1, 9, 10, 11, 13, 14, 15, 16, 2, 3, 4, 5, 6, 7, 8, 0, 24, 23, 22, 21, 20, 19, 18, 32,
                         31, 30, 29, 28, 27, 26, 25, 17]].astype(np.int32)
    
    outside = np.vstack((outside, forehead))

    regions = {"right_eyebrown": right_eyebrown, "left_eyebrown": left_eyebrown,
               "right_eye": right_eye, "left_eye": left_eye, "nose": nose, "between_eyes": between_eyes,
               "mouth": mouth, "chin": chin, "right_nasolabial_fold": right_nasolabial_fold, 
               "left_nasolabial_fold": left_nasolabial_fold, "right_cheek": right_cheek, 
               "left_cheek": left_cheek, "forehead": forehead, "outside": outside}

    region_masks = dict()
    for region_name, region_points in regions.items():
        # Create blank image
        mask = np.zeros((512, 512), dtype=np.uint8)

        # Fill with white pixels (255)
        cv2.fillPoly(mask, pts=[region_points], color=255)

        # If outside invert the mask
        if region_name == "outside":
            mask = cv2.bitwise_not(mask)

        # Normaliza to [0 - 1] range
        mask = (mask / 255).astype(np.float32)

        region_masks[region_name] = mask

    return region_masks
