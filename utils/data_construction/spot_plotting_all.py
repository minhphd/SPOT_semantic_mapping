import os
import json
import numpy as np
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import argparse

DEPTH_SCALE = 0.001

BODY_DEPTH_SOURCES = [
    "frontleft_depth_in_visual_frame",
    "frontright_depth_in_visual_frame",
    "left_depth_in_visual_frame",
    "right_depth_in_visual_frame",
    "back_depth_in_visual_frame",
]

HAND_DEPTH_SOURCE = "hand_depth_in_hand_color_frame"
HAND_COLOR_SOURCE = "hand_color_image"

def match_color_to_depth(rgb, depth):
    h, w = depth.shape
    if rgb.shape[:2] != (h,w):
        rgb = cv2.resize(rgb, (w,h))
    return rgb

def load_intrinsics(intr_path, cam):
    intr = json.load(open(intr_path))[cam]
    return (
        intr["focal_length"]["x"],
        intr["focal_length"]["y"],
        intr["principal_point"]["x"],
        intr["principal_point"]["y"],
    )

def load_odom_T_cam(meta_path):
    M = np.eye(4)
    meta = json.load(open(meta_path))
    T = meta["odom_T_cam"]
    pos = T["position"]
    rot = T["rotation"]

    M[:3,:3] = R.from_quat([rot["x"],rot["y"],rot["z"],rot["w"]]).as_matrix()
    M[:3,3] = [pos["x"],pos["y"],pos["z"]]
    return M

def make_pointcloud(depth, rgb, fx, fy, cx, cy):
    h, w = depth.shape
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))

    Z = depth.astype(np.float32) * DEPTH_SCALE
    X = (xs - cx) * Z / fx
    Y = (ys - cy) * Z / fy

    mask = (Z > 0.1) & (Z < 5.0)

    pts = np.stack([X[mask], Y[mask], Z[mask]], axis=-1)
    cols = rgb.reshape(-1,3)[mask.reshape(-1)] / 255.0

    return pts, cols

def fuse_dataset(dataset_dir):
    intr_path = os.path.join(dataset_dir, "intrinsics/intrinsics.json")
    meta_root = os.path.join(dataset_dir, "meta")

    pose_dir = os.path.join(meta_root, "robot_pose")
    timestamps = sorted(f[:-5] for f in os.listdir(pose_dir) if f.endswith(".json"))

    all_points = []
    all_colors = []

    for ts in timestamps:
        print(f"[INFO] Timestamp {ts}")

        # BODY cameras
        for cam in BODY_DEPTH_SOURCES:
            dp = os.path.join(dataset_dir, cam, ts + ".png")
            if not os.path.exists(dp):
                continue

            depth = cv2.imread(dp, -1)
            fx,fy,cx,cy = load_intrinsics(intr_path, cam)

            rgb_cam = cam.replace("depth_in_visual_frame", "fisheye_image")
            rp = os.path.join(dataset_dir, rgb_cam, ts + ".png")

            if os.path.exists(rp):
                rgb = cv2.cvtColor(cv2.imread(rp), cv2.COLOR_BGR2RGB)
                rgb = match_color_to_depth(rgb, depth)
            else:
                rgb = np.repeat((depth/depth.max())[...,None],3,axis=2)

            meta = os.path.join(meta_root, cam, ts + ".json")
            T = load_odom_T_cam(meta)

            pts, cols = make_pointcloud(depth, rgb, fx, fy, cx, cy)

            pts_h = np.c_[pts, np.ones(len(pts))]
            pts_w = (T @ pts_h.T).T[:, :3]

            all_points.append(pts_w)
            all_colors.append(cols)

        # HAND camera
        hd = os.path.join(dataset_dir, HAND_DEPTH_SOURCE, ts + ".png")
        hr = os.path.join(dataset_dir, HAND_COLOR_SOURCE, ts + ".png")
        hm = os.path.join(meta_root, HAND_DEPTH_SOURCE, ts + ".json")

        if os.path.exists(hd) and os.path.exists(hr) and os.path.exists(hm):
            depth = cv2.imread(hd, -1)
            rgb   = cv2.cvtColor(cv2.imread(hr), cv2.COLOR_BGR2RGB)

            fx,fy,cx,cy = load_intrinsics(intr_path, HAND_DEPTH_SOURCE)
            T = load_odom_T_cam(hm)

            pts, cols = make_pointcloud(depth, rgb, fx, fy, cx, cy)
            pts_h = np.c_[pts, np.ones(len(pts))]
            pts_w = (T @ pts_h.T).T[:, :3]

            all_points.append(pts_w)
            all_colors.append(cols)

    # Merge once
    P = np.vstack(all_points)
    C = np.vstack(all_colors)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P)
    pcd.colors = o3d.utility.Vector3dVector(C)

    pcd = pcd.voxel_down_sample(0.001)
    o3d.visualization.draw_geometries([pcd])

###############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data")
    args = parser.parse_args()
    fuse_dataset(args.dataset)
    