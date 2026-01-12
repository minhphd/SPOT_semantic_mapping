# utils/mask.py
"""
Mask utilities: mask merging, downsampling, IoU, large-structure filtering.
"""

import numpy as np
import cv2


def mask_iou(m1: np.ndarray, m2: np.ndarray) -> float:
    """Compute 2D IoU between two boolean masks."""
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    if union == 0:
        return 0.0
    return inter / union


def merge_overlapping_masks(masks, iou_thresh=0.5):
    """
    Merge overlapping masks (NMS-like behavior).
    Returns a reduced list of masks.
    """
    merged = []
    for m in masks:
        inserted = False
        for i in range(len(merged)):
            if mask_iou(m, merged[i]) > iou_thresh:
                merged[i] = np.logical_or(merged[i], m)
                inserted = True
                break
        if not inserted:
            merged.append(m)
    return merged


def downsample_mask(mask_full: np.ndarray,
                    target_h: int,
                    target_w: int) -> np.ndarray:
    """Downsample a full-resolution mask to depth resolution."""
    mask_small = cv2.resize(
        mask_full.astype(np.uint8),
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST
    )
    return mask_small.astype(bool)


def is_large_structure(mask_small,
                       pts_world,
                       img_frac_thresh=0.35,
                       extent_thresh=3.0):
    """
    Detect if object is a large structural surface (wall/floor).
    """
    h, w = mask_small.shape
    frac = mask_small.sum() / float(h * w)

    if frac < 0.01:
        return False

    if pts_world.shape[0] < 50:
        return False

    mins = pts_world.min(axis=0)
    maxs = pts_world.max(axis=0)
    extent = np.linalg.norm(maxs - mins)

    return (frac > img_frac_thresh) and (extent > extent_thresh)
