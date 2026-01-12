import os
import numpy as np
from tqdm import tqdm

np.float = np.float64
np.int = np.int_

import open3d as o3d
from PIL import Image
from argparse import ArgumentParser
from scipy.spatial.transform import Rotation
import skvideo.io
from utils.logger import load_full_tracker, build_logger
from utils.io import load_depth, load_conf, load_intrinsics, load_poses
o3d.visualization.webrtc_server.enable_webrtc()

# ============ CONFIG ============
DEPTH_WIDTH = 256
DEPTH_HEIGHT = 192
MAX_DEPTH = 20.0
RGB_WIDTH = 1920
RGB_HEIGHT = 1440
# ===============================

# ---------------------------------------
# Command-line args
# ---------------------------------------
def read_args():
    parser = ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("--every", type=int, default=5)
    parser.add_argument("--confidence", type=int, default=1)
    parser.add_argument("--tracker", type=str, required=True,
                        help="Path to saved tracker")
    return parser.parse_args()

# ---------------------------------------
# Build merged point cloud
# ---------------------------------------
def build_pointcloud(path, intr_o3d, poses, every=5, confidence_thresh=1):
    pc = o3d.geometry.PointCloud()

    video = skvideo.io.vreader(os.path.join(path, "rgb.mp4"))

    for i, (T_WC, rgb) in tqdm(enumerate(zip(poses, video))):
        if i % every != 0: continue

        conf = load_conf(os.path.join(path, "confidence", f"{i:06d}.png"))
        depth = load_depth(
            os.path.join(path, "depth", f"{i:06d}.png"),
            conf,
            filter_level=confidence_thresh
        )
        rgb = Image.fromarray(rgb).resize((DEPTH_WIDTH, DEPTH_HEIGHT))
        rgb = np.array(rgb)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb), o3d.geometry.Image(depth),
            depth_scale=1.0, depth_trunc=MAX_DEPTH, convert_rgb_to_intensity=False
        )

        T_CW = np.linalg.inv(T_WC)
        pc_i = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, intr_o3d, extrinsic=T_CW
        )
        pc += pc_i

    pc = pc.voxel_down_sample(0.02)
    return pc

# ---------------------------------------
# Convert tracker objects → bounding boxes
# ---------------------------------------
def tracker_boxes_to_o3d(tracker):
    boxes = []
    for obj in tracker.objects:
        boxes.append(obj.bbox)
    return boxes

# ---------------------------------------
# MAIN
# ---------------------------------------
def main():
    flags = read_args()

    # 1. Load dataset
    intrinsics = load_intrinsics(
        os.path.join(flags.path, "camera_matrix.csv"), 
        scale_x=DEPTH_WIDTH/RGB_WIDTH, 
        scale_y=DEPTH_HEIGHT/RGB_HEIGHT
        )  # dict with {"fx", "fy", "cx", "cy"}
    
    o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=DEPTH_WIDTH, 
        height=DEPTH_HEIGHT, 
        fx=intrinsics["fx"], 
        fy=intrinsics["fy"], 
        cx=intrinsics["cx"], 
        cy=intrinsics["cy"]
    )

    poses = load_poses(os.path.join(flags.path, "odometry.csv"))

    # 2. Build original pointcloud
    pc = build_pointcloud(flags.path, o3d_intrinsics, poses,
                          every=flags.every,
                          confidence_thresh=flags.confidence)

    print("Loaded point cloud with", np.asarray(pc.points).shape[0], "points.")

    # 3. Load tracker
    logger = build_logger("")
    tracker = load_full_tracker(flags.tracker ,logger)[0]

    o3d.visualization.draw_geometries([pc] + tracker_boxes_to_o3d(tracker))
    
if __name__ == "__main__":
    main()
