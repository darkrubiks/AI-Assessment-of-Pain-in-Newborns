"""
attribution_mask_processing.py

Author: Leonardo Antunes Ferreira
Date:20/02/2023

TODO

"""
import cv2
import numpy as np
from sklearn.cluster import KMeans


def attribution_mask_processing(attribution_mask, n_clusters=5, ksize=11, sigma=0, alpha_thres=0.5):
        
    if len(attribution_mask.shape) <= 2:
        attribution_mask = np.expand_dims(attribution_mask, axis=-1)

    kmeans = KMeans(n_clusters=n_clusters, random_state=123)
    kmeans.fit(attribution_mask.reshape(-1, 1))

    center = kmeans.cluster_centers_ # pegando os 3 centros
    res = center[kmeans.labels_.flatten()] # atribuindo o respectivo centro para cada pixel da imagem
    res2 = res.reshape((attribution_mask.shape)) # vetor para formato de imagem
    res2 = np.squeeze(res2)
    res2 = cv2.GaussianBlur(res2,(ksize,ksize), sigma) # filtro gaussiano

    alpha_channel = np.ones(np.squeeze(res2).shape, dtype=res2.dtype)
    alpha_channel[np.where(res2 <= res2.mean() + res2.std()*alpha_thres)[:2]] = 0

    return res2, alpha_channel

