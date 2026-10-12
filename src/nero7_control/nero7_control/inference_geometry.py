"""Distortion-aware optical rays used by the grasp network geometry."""
import cv2
import numpy as np


def optical_rays(pixels, K, distortion=(), distortion_model='plumb_bob'):
    K = np.asarray(K, float)
    pixels = np.asarray(pixels, float).reshape(-1, 2)
    coefficients = np.asarray(distortion, float)
    if (K.shape != (3, 3) or not np.isfinite(K).all()
            or K[0, 0] <= 0 or K[1, 1] <= 0 or not np.isfinite(coefficients).all()):
        raise ValueError('Invalid optical intrinsics')
    if coefficients.size and np.any(coefficients):
        if distortion_model == 'equidistant' and coefficients.size == 4:
            xy = cv2.fisheye.undistortPoints(pixels.reshape(-1, 1, 2), K, coefficients)
        elif distortion_model in ('plumb_bob', 'rational_polynomial') and coefficients.size in (4, 5, 8, 12, 14):
            xy = cv2.undistortPoints(pixels.reshape(-1, 1, 2), K, coefficients)
        else:
            raise ValueError('Unsupported optical distortion')
        xy = xy.reshape(-1, 2)
    else:
        xy = (pixels-K[:2, 2])/[K[0, 0], K[1, 1]]
    return np.column_stack([xy, np.ones(len(xy))])
