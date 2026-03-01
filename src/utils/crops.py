# utils/crops.py
"""
Crop utilities: padding and horizontal concatenation for multi-view captioning.
"""

import numpy as np
import cv2
import os


def concat_crops_horizontal(crops):
    """
    Concatenate multiple RGB crops horizontally with padding.

    Parameters
    ----------
    crops : list[np.ndarray]
        List of RGB crops.

    Returns
    -------
    np.ndarray or None
        Fused image or None if no crops.
    """
    if len(crops) == 0:
        return None
    if len(crops) == 1:
        return crops[0]

    max_h = max(c.shape[0] for c in crops)
    max_w = max(c.shape[1] for c in crops)

    padded = []
    for c in crops:
        h, w = c.shape[:2]
        pad_h = max_h - h
        pad_w = max_w - w

        c_pad = np.pad(
            c,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode='constant',
            constant_values=0
        )
        padded.append(c_pad)

    fused = np.concatenate(padded, axis=1)
    return fused
