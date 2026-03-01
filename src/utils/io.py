"""
I/O utilities for loading camera intrinsics, poses, depth maps, and confidence maps.
"""

import os
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import pandas as pd
import open3d as o3d

def load_intrinsics(path: str, scale_x=1, scale_y=1):
    """
    Load camera intrinsics from a CSV file.

    Parameters
    ----------
    path : str
        Path to CSV with fx, fy, cx, cy in camera_matrix.csv format.

    Returns
    -------
    dict
        Dictionary with {"fx", "fy", "cx", "cy"} in depth resolution.
    """
    cam = np.loadtxt(path, delimiter=",")
    fx, fy = cam[0, 0] * scale_x, cam[1, 1] * scale_y
    cx, cy = cam[0, 2] * scale_x, cam[1, 2] * scale_y
    
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy}


def load_poses(path: str):
    """
    Load odometry poses (world_T_cam) from a CSV file using pandas.

    CSV format:
        timestamp, frame, x, y, z, qx, qy, qz, qw

    Parameters
    ----------
    path : str
        Path to the odometry CSV file.

    Returns
    -------
    
    """
    odometry = np.loadtxt(path, delimiter=',', skiprows=1)
    poses = []

    for line in odometry:
        # timestamp, frame, x, y, z, qx, qy, qz, qw
        position = line[2:5]
        quaternion = line[5:]
        T_WC = np.eye(4)
        T_WC[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
        T_WC[:3, 3] = position
        poses.append(T_WC)
    return poses


def load_depth(path: str, confidence=None, filter_level=0):
    """
    Load a depth map from .npy or .png.

    Depth is assumed to be in millimeters stored as uint16 or npy.

    Returns
    -------
    np.ndarray
        Depth map in meters (float32).
    """
    if path.endswith(".npy"):
        depth_mm = np.load(path, mmap_mode="r")
    else:
        depth_mm = np.array(Image.open(path))    
    depth_m = depth_mm.astype(np.float32) / 1000.0
    if confidence is not None:
        depth_m[confidence < filter_level] = 0.0
    return depth_m


def load_conf(path: str):
    """
    Load a confidence map if available.

    Returns
    -------
    np.ndarray
        Confidence image as uint8 or None if missing.
    """
    if not os.path.exists(path):
        return None
    return np.array(Image.open(path))
