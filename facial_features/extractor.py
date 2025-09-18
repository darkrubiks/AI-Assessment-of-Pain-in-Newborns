import numpy as np


def euclidean_dist(p1, p2):
    return np.linalg.norm(p1 - p2)


def compute_ear(eye_pts):
    p1, p2, p3, p4, p5, p6 = eye_pts

    vertical_1 = euclidean_dist(p2, p6)
    vertical_2 = euclidean_dist(p3, p5)
    horizontal = euclidean_dist(p1, p4)

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)

    return ear


def compute_mar(mouth_pts):
    p1, p2, p3, p4, p5, p6, p7, p8 = mouth_pts

    vertical_1 = euclidean_dist(p2, p8)
    vertical_2 = euclidean_dist(p3, p7)
    vertical_3 = euclidean_dist(p4, p6)
    horizontal = euclidean_dist(p1, p5)

    mar = (vertical_1 + vertical_2 + vertical_3) / (2.0 * horizontal)

    return mar