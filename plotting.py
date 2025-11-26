import os
import json
import numpy as np
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation as R


###############################################################################
# Helpers
###############################################################################

def match_color_to_depth(rgb, depth):
    """Ensure RGB matches depth resolution."""
    h, w = depth.shape
    if rgb.shape[:2] != (h, w):
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    return rgb


def load_intrinsics(intr_path, cam_src):
    with open(intr_path, "r") as f:
        intr_all = json.load(f)
    intr = intr_all[cam_src]
    fx = intr["focal_length"]["x"]
    fy = intr["focal_length"]["y"]
    cx = intr["principal_point"]["x"]
    cy = intr["principal_point"]["y"]
    return fx, fy, cx, cy


def load_odom_T_cam(meta_path):
    """Load odom_T_cam from new capture metadata."""
    with open(meta_path, "r") as f:
        meta = json.load(f)
    T = meta["odom_T_cam"]
    pos = T["position"]
    rot = T["rotation"]

    M = np.eye(4)
    M[:3, :3] = R.from_quat([rot["x"], rot["y"], rot["z"], rot["w"]]).as_matrix()
    M[:3, 3] = [pos["x"], pos["y"], pos["z"]]
    return M


def make_pointcloud(depth, rgb, fx, fy, cx, cy):
    """Convert RGB + depth to Nx3 points + colors."""
    h, w = depth.shape

    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    Z = depth.astype(np.float32)
    X = (xs - cx) * Z / fx
    Y = (ys - cy) * Z / fy

    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    colors = rgb.reshape(-1, 3) / 255.0

    # remove invalid depth
    mask = Z.reshape(-1) > 0
    return pts[mask], colors[mask]


###############################################################################
# Main function
###############################################################################

def visualize_timestamp(dataset_dir, timestamp):
    intr_path = os.path.join(dataset_dir, "intrinsics", "intrinsics.json")
    meta_root = os.path.join(dataset_dir, "meta")

    pcd_all = []

    # Only use body depth cameras as geometry sources
    depth_cams = [
        d for d in os.listdir(dataset_dir)
        if d.endswith("_depth_in_visual_frame")
        and os.path.isdir(os.path.join(dataset_dir, d))
    ]

    if not depth_cams:
        print("[ERROR] No depth cameras found in dataset_dir.")
        return

    # ---------------------------
    # BODY CAMERAS
    # ---------------------------
    for depth_cam in depth_cams:
        if "hand" in depth_cam:
            # body only here; hand is handled separately below
            continue

        depth_dir = os.path.join(dataset_dir, depth_cam)
        depth_path = os.path.join(depth_dir, f"{timestamp}.png")
        if not os.path.exists(depth_path):
            continue

        # Metadata for this depth camera
        meta_path = os.path.join(meta_root, depth_cam, f"{timestamp}.json")
        if not os.path.exists(meta_path):
            print(f"[WARN] No meta for {depth_cam} at {timestamp}, skipping.")
            continue

        print(f"[INFO] Loading depth from {depth_cam}")

        # Load depth
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None or depth.dtype != np.uint16:
            print(f"[WARN] Depth image invalid for {depth_cam}, skipping.")
            continue

        # Find matching RGB camera (body cameras use fisheye_image)
        rgb_cam = depth_cam.replace("depth_in_visual_frame", "fisheye_image")
        rgb_dir = os.path.join(dataset_dir, rgb_cam)
        rgb_path = os.path.join(rgb_dir, f"{timestamp}.png")

        if os.path.exists(rgb_path):
            rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
            rgb = match_color_to_depth(rgb, depth)
            print(f"[INFO] Using RGB from {rgb_cam}")
        else:
            # Fallback: grayscale from depth
            print(f"[WARN] No RGB found for {depth_cam}, using grayscale.")
            rgb = np.repeat((depth / max(depth.max(), 1e-6))[..., None], 3, axis=2)

        # Load intrinsics for the depth camera
        try:
            fx, fy, cx, cy = load_intrinsics(intr_path, depth_cam)
        except KeyError:
            print(f"[WARN] No intrinsics for {depth_cam}, skipping.")
            continue

        # Load camera pose in odom frame
        T_odom_cam = load_odom_T_cam(meta_path)

        # Build point cloud in camera frame
        pts, cols = make_pointcloud(depth, rgb, fx, fy, cx, cy)

        # Transform into odom frame
        pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
        pts_world = (T_odom_cam @ pts_h.T).T[:, :3]

        # Construct Open3D cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_world)
        pcd.colors = o3d.utility.Vector3dVector(cols)
        pcd_all.append(pcd)

    # ---------------------------
    # HAND CAMERA
    # ---------------------------

    # Use depth already aligned to color frame
    hand_depth_cam = "hand_depth_in_hand_color_frame"
    hand_color_cam = "hand_color_image"  # aligned RGB

    depth_dir = os.path.join(dataset_dir, hand_depth_cam)
    color_dir = os.path.join(dataset_dir, hand_color_cam)

    depth_path = os.path.join(depth_dir, f"{timestamp}.png")
    color_path = os.path.join(color_dir, f"{timestamp}.png")

    meta_path = os.path.join(meta_root, hand_depth_cam, f"{timestamp}.json")

    # Check existence
    if os.path.exists(depth_path) and os.path.exists(color_path) and os.path.exists(meta_path):
        print("[INFO] Loading HAND camera")

        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        rgb = cv2.cvtColor(cv2.imread(color_path), cv2.COLOR_BGR2RGB)
        rgb = match_color_to_depth(rgb, depth)
        print(rgb.shape, depth.shape)

        if depth is None or rgb is None or depth.dtype != np.uint16:
            print("[WARN] Hand images unavailable or corrupted.")
        else:
            # Intrinsics
            try:
                fx, fy, cx, cy = load_intrinsics(intr_path, hand_depth_cam)
            except KeyError:
                print("[WARN] No intrinsics for hand camera")
                fx = fy = cx = cy = None

            # Hand color + depth already same resolution, so no resize needed
            # Extrinsics: directly use odom_T_cam from metadata
            T_odom_cam = load_odom_T_cam(meta_path)

            # Generate point cloud
            pts, cols = make_pointcloud(depth, rgb, fx, fy, cx, cy)

            # Transform to odom frame
            pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
            pts_world = (T_odom_cam @ pts_h.T).T[:, :3]

            # Construct Open3D cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts_world)
            pcd.colors = o3d.utility.Vector3dVector(cols)

            pcd_all.append(pcd)
    else:
        print("[INFO] No hand camera data for this timestamp.")

    # ---------------------------
    # Merge all point clouds
    # ---------------------------

    if len(pcd_all) == 0:
        print("[ERROR] No valid point clouds found.")
        return

    merged = pcd_all[0]
    for p in pcd_all[1:]:
        merged += p

    # Optional downsample
    merged = merged.voxel_down_sample(voxel_size=0.05)

    # Visualize
    print("[INFO] Visualizing fused point cloud")
    o3d.visualization.draw_geometries([merged])


###############################################################################
# Example usage
###############################################################################

if __name__ == "__main__":
    visualize_timestamp(
        dataset_dir="data", 
        timestamp="20251116_133227",
    )
