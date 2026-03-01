# utils/geometry.py
"""
Geometric utilities for projecting masks into 3D and normalizing vectors.
"""

import numpy as np
import open3d.t as o3d
from collections import Counter

def apply_dbscan(pcd, eps=0.5, min_samples=10): # borrowed from ConceptGraph (words for words)
    pcd_clusters = pcd.cluster_dbscan(
        eps=eps,
        min_points=min_samples,
    )
    
    # Convert to numpy arrays
    obj_points = np.asarray(pcd.points)
    obj_colors = np.asarray(pcd.colors)
    pcd_clusters = np.array(pcd_clusters)

    # Count all labels in the cluster
    counter = Counter(pcd_clusters)

    # Remove the noise label
    if counter and (-1 in counter):
        del counter[-1]

    if counter:
        # Find the label of the largest cluster
        most_common_label, _ = counter.most_common(1)[0]
        
        # Create mask for points in the largest cluster
        largest_mask = pcd_clusters == most_common_label

        # Apply mask
        largest_cluster_points = obj_points[largest_mask]
        largest_cluster_colors = obj_colors[largest_mask]
        
        # If the largest cluster is too small, return the original point cloud
        if len(largest_cluster_points) < 5:
            return pcd

        # Create a new PointCloud object
        largest_cluster_pcd = o3d.geometry.PointCloud()
        largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)
        largest_cluster_pcd.colors = o3d.utility.Vector3dVector(largest_cluster_colors)
        
        pcd = largest_cluster_pcd
    
    return pcd

import numpy as np
import open3d as o3d

def construct_top_down(
    rgb,                 # (H, W, 3) uint8
    depth,               # (H, W) float32
    T_WC,                # (4,4) camera->world transform
    o3d_intrinsics,      # open3d.camera.PinholeCameraIntrinsic
    max_depth,           # float
    canvas,              # dict storing (img, x_min, z_min, res)
    up_axis="y",         # "y" = Y-up (use X,Z), "z" = Z-up (use X,Y)
    voxel_size=0.03     # e.g. 0.03 to downsample before painting
):
    """
    Builds a top-down RGB map and updates the given canvas.

    canvas must contain:
      canvas['img']   : (Hc, Wc, 3) float32 accumulator
      canvas['count'] : (Hc, Wc)    int32   — number of points per cell
      canvas['x_min'] : float — global min X
      canvas['z_min'] : float — global min Z
      canvas['res']   : float — meters per pixel

    Returns the updated canvas.
    """
    
    assert rgb.shape[:2] == depth.shape, "RGB and depth must have the same resolution."
    # project depth + rgb into 3D point cloud
    depth = np.ascontiguousarray(depth)
    depth_o3d = o3d.geometry.Image(depth.astype(np.float32))
    rgb_o3d   = o3d.geometry.Image(rgb.astype(np.uint8))

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d,
        depth_o3d,
        depth_scale=1.0,
        depth_trunc=max_depth,
        convert_rgb_to_intensity=False
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d_intrinsics)

    # camera → world
    pcd.transform(T_WC)

    if voxel_size is not None:
        pcd = pcd.voxel_down_sample(voxel_size)

    if len(pcd.points) == 0:
        return canvas

    XYZ = np.asarray(pcd.points)              # (N, 3)
    RGB = np.asarray(pcd.colors, float)       # (N, 3), [0,1]

    # --------------------------------------------------------
    # 2. Choose top-down plane based on up_axis
    # --------------------------------------------------------
    if up_axis == "y":          # typical: Y is height
        Xp = XYZ[:, 0]          # left/right
        Zp = XYZ[:, 2]          # forward
    elif up_axis == "z":        # if Z is height in your data
        Xp = XYZ[:, 0]
        Zp = XYZ[:, 1]
    else:
        raise ValueError(f"Unsupported up_axis '{up_axis}', use 'y' or 'z'.")

    # --------------------------------------------------------
    # 3. Map world coords → canvas indices
    # --------------------------------------------------------
    x_min = canvas["x_min"]
    z_min = canvas["z_min"]
    res   = canvas["res"]

    img = canvas["img"]      # (Hc, Wc, 3) float32 accumulator
    cnt = canvas["count"]    # (Hc, Wc) int32

    Hc, Wc, _ = img.shape

    px = ((Xp - x_min) / res).astype(np.int32)
    pz = ((Zp - z_min) / res).astype(np.int32)

    # Check if any points are outside the canvas bounds
    x_max = x_min + Wc * res
    z_max = z_min + Hc * res

    expand_needed = False
    if np.any(Xp < x_min) or np.any(Xp >= x_max) or np.any(Zp < z_min) or np.any(Zp >= z_max):
        expand_needed = True

    if expand_needed:
        # Calculate new bounds
        new_x_min = min(x_min, np.min(Xp))
        new_x_max = max(x_max, np.max(Xp))
        new_z_min = min(z_min, np.min(Zp))
        new_z_max = max(z_max, np.max(Zp))

        # Calculate new canvas size
        new_Wc = int(np.ceil((new_x_max - new_x_min) / res))
        new_Hc = int(np.ceil((new_z_max - new_z_min) / res))

        # Create new canvas
        new_img = np.zeros((new_Hc, new_Wc, 3), dtype=np.float32)
        new_cnt = np.zeros((new_Hc, new_Wc), dtype=np.int32)

        # Copy old canvas into new canvas
        x_offset = int((x_min - new_x_min) / res)
        z_offset = int((z_min - new_z_min) / res)

        new_img[z_offset:z_offset + Hc, x_offset:x_offset + Wc] = img
        new_cnt[z_offset:z_offset + Hc, x_offset:x_offset + Wc] = cnt

        # Update canvas
        canvas["img"] = new_img
        canvas["count"] = new_cnt
        canvas["x_min"] = new_x_min
        canvas["z_min"] = new_z_min

        img = new_img
        cnt = new_cnt
        Hc, Wc, _ = img.shape

        px = ((Xp - new_x_min) / res).astype(np.int32)
        pz = ((Zp - new_z_min) / res).astype(np.int32)

    valid = (px >= 0) & (px < Wc) & (pz >= 0) & (pz < Hc)
    if not np.any(valid):
        return canvas

    px = px[valid]
    pz = pz[valid]
    colors = RGB[valid]      # (M,3)

    # --------------------------------------------------------
    # 4. Vectorized accumulation with np.add.at
    # --------------------------------------------------------
    flat_img = img.reshape(-1, 3)
    flat_cnt = cnt.ravel()

    idx = pz * Wc + px       # (M,)

    # accumulate each channel
    np.add.at(flat_img[:, 0], idx, colors[:, 0])
    np.add.at(flat_img[:, 1], idx, colors[:, 1])
    np.add.at(flat_img[:, 2], idx, colors[:, 2])
    np.add.at(flat_cnt,       idx, 1)

    # write back
    canvas["img"]   = flat_img.reshape(Hc, Wc, 3)
    canvas["count"] = flat_cnt.reshape(Hc, Wc)

    return canvas

def finalize_canvas(canvas):
    img = canvas["img"]
    cnt = canvas["count"]

    valid = cnt > 0
    out = np.zeros_like(img, dtype=np.float32)
    out[valid] = img[valid] / cnt[valid][..., None]

    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return out

def init_tsdf_canvas(voxel_size=0.03, sdf_trunc=0.09):
    """
    Create a TSDF volume to accumulate all RGB-D frames.
    """
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    return {"volume": volume}

def tsdf_integrate_frame(
    rgb,               # (H, W, 3) uint8, already aligned to depth
    depth,             # (H, W) float32, meters
    T_WC,              # (4,4) camera->world
    o3d_intrinsics,    # PinholeCameraIntrinsic
    max_depth,         # depth_trunc
    tsdf_canvas,
):
    """
    Integrate one RGB-D frame into the TSDF volume.
    """
    volume = tsdf_canvas["volume"]

    assert rgb.shape[:2] == depth.shape, "RGB and depth must match."

    depth_o3d = o3d.geometry.Image(depth.astype(np.float32))
    rgb_o3d   = o3d.geometry.Image(rgb.astype(np.uint8))

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d,
        depth_o3d,
        depth_scale=1.0,
        depth_trunc=max_depth,
        convert_rgb_to_intensity=False,
    )

    # Open3D expects extrinsic: world -> camera
    extrinsic = np.linalg.inv(T_WC)

    volume.integrate(
        rgbd,
        o3d_intrinsics,
        extrinsic,
    )

def extract_horizontal_surfaces(pcd, up_axis="y", threshold_deg=20):
    """
    Filters point cloud keeping only points with normals
    nearly vertical (horizontal surfaces).

    Args:
        pcd: open3d.geometry.PointCloud with normals estimated.
        up_axis: "y" (default) or "z" depending on your world frame.
        threshold_deg: max angular deviation from vertical.
    """
    # Pick up vector
    if up_axis == "y":
        up = np.array([0, 1, 0], dtype=np.float32)
    elif up_axis == "z":
        up = np.array([0, 0, 1], dtype=np.float32)
    else:
        raise ValueError("up_axis must be 'y' or 'z'")

    # Extract normals
    normals = np.asarray(pcd.normals)  # (N,3)

    # Dot product gives cos(angle)
    dot = normals @ up
    dot = np.clip(dot, -1, 1)

    # Convert threshold_deg → cosine threshold
    angle = np.radians(threshold_deg)
    cos_th = np.cos(angle)

    # Keep normals within ±threshold of vertical
    mask = np.abs(dot) >= cos_th   # horizontal surfaces

    idx = np.where(mask)[0]
    return pcd.select_by_index(idx)

def tsdf_extract_topdown(
    tsdf_canvas,
    res=0.03,      # meters per pixel
    up_axis="y",   # "y" if Y is up, else "z"
):
    """
    Extract a fused top-down RGB map from the TSDF volume.
    Returns:
        topdown_rgb: (Hc, Wc, 3) uint8
    """
    volume = tsdf_canvas["volume"]

    # fused geometry
    pcd = volume.extract_point_cloud()
    
    if len(pcd.points) == 0:
        raise RuntimeError("TSDF volume is empty - no points to project.")

    XYZ = np.asarray(pcd.points)        # (N, 3)
    RGB = np.asarray(pcd.colors)        # (N, 3), [0,1]

    if up_axis == "y":          # Y-up → use X,Z
        Xp = XYZ[:, 0]
        Zp = XYZ[:, 2]
    elif up_axis == "z":        # Z-up → use X,Y
        Xp = XYZ[:, 0]
        Zp = XYZ[:, 1]
    else:
        raise ValueError(f"Unsupported up_axis '{up_axis}'")

    x_min, x_max = Xp.min(), Xp.max()
    z_min, z_max = Zp.min(), Zp.max()

    Wc = int(np.ceil((x_max - x_min) / res)) + 1
    Hc = int(np.ceil((z_max - z_min) / res)) + 1

    img  = np.zeros((Hc, Wc, 3), dtype=np.float32)
    cnt  = np.zeros((Hc, Wc), dtype=np.int32)

    px = ((Xp - x_min) / res).astype(np.int32)
    pz = ((Zp - z_min) / res).astype(np.int32)

    valid = (px >= 0) & (px < Wc) & (pz >= 0) & (pz < Hc)
    px = px[valid]
    pz = pz[valid]
    colors = RGB[valid]

    flat_img = img.reshape(-1, 3)
    flat_cnt = cnt.ravel()
    idx = pz * Wc + px

    np.add.at(flat_img[:, 0], idx, colors[:, 0])
    np.add.at(flat_img[:, 1], idx, colors[:, 1])
    np.add.at(flat_img[:, 2], idx, colors[:, 2])
    np.add.at(flat_cnt,       idx, 1)

    img  = flat_img.reshape(Hc, Wc, 3)
    cnt  = flat_cnt.reshape(Hc, Wc)

    mask = cnt > 0
    img[mask] /= cnt[mask][..., None]

    topdown_rgb = (img * 255.0).clip(0, 255).astype(np.uint8)
    return topdown_rgb
