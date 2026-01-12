# utils/tsdf.py
"""
TSDF utilities: initialization and per-frame integration.
"""

import numpy as np
import open3d as o3d


def init_tsdf(intr, depth_width, depth_height,
              voxel=0.02, trunc=0.06):
    """
    Create and configure a Scalable TSDF Volume.

    Returns
    -------
    tsdf_volume : open3d TSDF volume
    intrinsic_o3d : Pinhole intrinsics for Open3D
    """
    intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic()
    intrinsic_o3d.set_intrinsics(
        depth_width,
        depth_height,
        intr["fx"], intr["fy"],
        intr["cx"], intr["cy"],
    )

    tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel,
        sdf_trunc=trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    return tsdf_volume, intrinsic_o3d


def integrate_tsdf(volume,
                   rgb: np.ndarray,
                   depth: np.ndarray,
                   intrinsic_o3d,
                   extrinsic,
                   depth_trunc=3.0):
    """
    Integrate a frame into the TSDF volume.
    """
    depth_o3d = o3d.geometry.Image((depth * 1000.0).astype(np.uint16))
    color_o3d = o3d.geometry.Image(rgb.astype(np.uint8))

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        depth_scale=1000.0,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False
    )

    volume.integrate(
        rgbd,
        intrinsic_o3d,
        extrinsic
    )
