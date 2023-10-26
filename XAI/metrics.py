import numpy as np


def calculate_xai_score(image, region_masks):
    region_scores = {}
    for region, region_mask in region_masks.items():
        region_importance = np.mean(image * region_mask)
        region_scores[region] = region_importance

    total_score = sum(region_scores.values())
    
    normalized_scores = {}
    for region, score in region_scores.items():
        normalized_score = score / total_score
        normalized_scores[region] = normalized_score
    
    return normalized_scores

