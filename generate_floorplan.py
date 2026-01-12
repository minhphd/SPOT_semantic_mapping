import os
import open3d as o3d
import numpy as np

np.int = np.int_
np.float = np.float64

from scipy.spatial.transform import Rotation
from argparse import ArgumentParser
from PIL import Image
import skvideo.io
from tqdm import tqdm
from utils.io import load_intrinsics, load_poses, load_depth, load_conf
from configs.loader import cfg

# How large the floor plan grid is
RESOLUTION = 0.05   # meters per pixel (5cm)
NORMAL_UP = np.array([0, 1, 0])   # Y-up for Stray Scanner

# ============================================================
# CLI
# ============================================================
description = "Generate a 2D floor plan from StrayScanner dataset"
usage = "python generate_floorplan.py <dataset-folder>"

def read_args():
    parser = ArgumentParser(description=description, usage=usage)
    parser.add_argument('path', type=str)
    parser.add_argument('--every', type=int, default=3)
    parser.add_argument('--confidence', '-c', type=int, default=1)
    
    # RANSAC / wall params (tweak if needed)
    parser.add_argument('--plane_dist_thresh', type=float, default=0.05) # How far points can be from plane to be inlier
    parser.add_argument('--plane_min_points', type=int, default=800) # minimum inlier points to accept a plane
    parser.add_argument('--max_planes', type=int, default=100)  # maximum number of planes to extract
    return parser.parse_args()

# ============================================================
# Utility
# ============================================================
# def _resize_camera_matrix(M, sx, sy):
#     fx, fy = M[0,0], M[1,1]
#     cx, cy = M[0,2], M[1,2]
#     return np.array([
#         [fx*sx,   0,    cx*sx],
#         [0,     fy*sy, cy*sy],
#         [0,        0,      1]
#     ])

# def load_depth(path, confidence=None, filter_level=0):
#     if path.endswith(".npy"):
#         depth_mm = np.load(path)
#     else:
#         depth_mm = np.array(Image.open(path))
#     depth_m = depth_mm.astype(np.float32) / 1000.0
#     if confidence is not None:
#         depth_m[confidence < filter_level] = 0
#     return o3d.geometry.Image(depth_m)

# def get_intrinsics(intr):
#     scaled = _resize_camera_matrix(intr, DEPTH_WIDTH/1920, DEPTH_HEIGHT/1440)
#     return o3d.camera.PinholeCameraIntrinsic(
#         width=DEPTH_WIDTH, height=DEPTH_HEIGHT,
#         fx=scaled[0,0], fy=scaled[1,1],
#         cx=scaled[0,2], cy=scaled[1,2]
#     )

# ============================================================
# LOAD DATA
# ============================================================
def read_data(flags):
    intrinsics = np.loadtxt(os.path.join(flags.path, 'camera_matrix.csv'), delimiter=',')
    odom = np.loadtxt(os.path.join(flags.path, 'odometry.csv'), delimiter=',', skiprows=1)

    poses = []
    for line in odom:
        T = np.eye(4)
        T[:3,:3] = Rotation.from_quat(line[5:]).as_matrix()
        T[:3,3] = line[2:5]
        poses.append(T)

    depth_dir = os.path.join(flags.path, "depth")
    depth_frames = sorted(
        [os.path.join(depth_dir,f) for f in os.listdir(depth_dir)
         if f.endswith(".png") or f.endswith(".npy")]
    )

    return {
        "poses": poses,
        "intrinsics": intrinsics,
        "depth_frames": depth_frames
    }

# ============================================================
# FLOOR PLAN BUILDER (with plane RANSAC)
# ============================================================
def generate_floor_plan(flags):
    DEPTH_WIDTH = cfg.camera["depth_width"]
    DEPTH_HEIGHT = cfg.camera["depth_height"]
    RGB_WIDTH = cfg.camera["rgb_width"]
    RGB_HEIGHT = cfg.camera["rgb_height"]
    OUTPUT_DIR = cfg.pipeline.get("output_dir", "outputs")
    # EXP_NAME = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
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

    rgb_path = os.path.join(flags.path, "rgb.mp4")
    video = skvideo.io.vreader(rgb_path)
    
    poses = load_poses(os.path.join(flags.path, "odometry.csv"))
    depth_path = os.path.join(flags.path, "depth")
    confidence_path = os.path.join(flags.path, "confidence")
    rgb_path = os.path.join(flags.path, "rgb.mp4")

    # collect 3D candidate wall points across all frames
    wall_points_3d = []

    for i, (T_WC, rgb) in tqdm(enumerate(zip(poses, video))):
        if i % flags.every != 0:
            continue

        # print(f"Processing frame {i}", end="\r")
        confidence = load_conf(os.path.join(confidence_path, f"{i:06d}.png"))
        depth = load_depth(
            os.path.join(depth_path, f"{i:06d}.png"),
            confidence,
            filter_level=cfg.projection["min_confidence"]
        )
        # RGB resize
        rgb = Image.fromarray(rgb)
        rgb = np.array(rgb.resize((DEPTH_WIDTH, DEPTH_HEIGHT)))

        # Create RGBD
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb), o3d.geometry.Image(depth),
            depth_scale=1.0, depth_trunc=cfg.camera['max_depth'], convert_rgb_to_intensity=False
        )

        # Convert to PCD (in camera frame)
        T_CW = np.linalg.inv(T_WC)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d_intrinsics, extrinsic=T_CW)

        if len(pcd.points) == 0:
            continue

        # Estimate surface normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30)
        )

        pts = np.asarray(pcd.points)
        nrm = np.asarray(pcd.normals)

        # Reject very low (floor) and very high (ceiling) points.
        # In StrayScanner, Y is the vertical axis (up), so:
        #   - small Y  → near the floor
        #   - large Y  → near the ceiling
        # Keep only points within a plausible wall height band.
        y = pts[:,1]
        height_mask = (y > 0.2) & (y < 2.5) # assuming ceiling is above 2.5m

        # Classify walls based on normals:
        # NORMAL_UP = [0, 1, 0] is the "up" direction.
        # For a vertical wall, its normal should be close to horizontal,
        # i.e., nearly orthogonal to NORMAL_UP.
        #
        # cos_sim = |n ⋅ NORMAL_UP|:
        #   - cos_sim ≈ 1 → normal is vertical (floor/ceiling)
        #   - cos_sim ≈ 0 → normal is horizontal (wall)
        cos_sim = np.abs(nrm @ NORMAL_UP)

        # Threshold to keep only near-horizontal normals as wall candidates.
        # 0.3 is somewhat tolerant to noise / imperfect normals.
        vertical_mask = cos_sim < 0.3

        # Final wall mask: points that are within the height band
        # AND whose normals indicate a vertical surface.
        wall_mask = height_mask & vertical_mask

        # Extract 3D coordinates of candidate wall points for this frame.
        wall_pts = pts[wall_mask]

        # Only append if we actually found any wall points.
        if wall_pts.size > 0:
            wall_points_3d.append(wall_pts)

    print()  # newline after progress

    # ===========================
    # RANSAC WALL PLANE CLEANING
    # ===========================
    if len(wall_points_3d) == 0:
        print("No wall candidate points collected!")
        return

    walls_all = np.vstack(wall_points_3d)

    # build Open3D cloud
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(walls_all)

    # downsample for speed
    cloud = cloud.voxel_down_sample(voxel_size=0.01)

    remaining = cloud
    plane_points = []

    print("Running RANSAC plane segmentation...")
    for k in range(flags.max_planes):
        if len(remaining.points) < flags.plane_min_points:
            break

        plane_model, inliers = remaining.segment_plane(
            distance_threshold=flags.plane_dist_thresh,
            ransac_n=3,
            num_iterations=2000
        )

        if len(inliers) < flags.plane_min_points:
            break

        plane = remaining.select_by_index(inliers)
        plane_points.append(np.asarray(plane.points))

        # remove inliers and continue
        remaining = remaining.select_by_index(inliers, invert=True)
        print(f"  Plane {k}: {len(inliers)} pts, remaining {len(remaining.points)}")

    if len(plane_points) == 0:
        print("RANSAC found no strong wall planes; falling back to raw walls.")
        final_walls = np.asarray(cloud.points)
    else:
        final_walls = np.vstack(plane_points)

    # ===========================
    # BUILD FLOOR PLAN GRID
    # ===========================
    # project to XZ (Y is up)
    walls_xz = final_walls[:, [0, 2]]

    xmin, zmin = walls_xz.min(axis=0)
    xmax, zmax = walls_xz.max(axis=0)

    w = int((xmax - xmin) / RESOLUTION) + 10
    h = int((zmax - zmin) / RESOLUTION) + 10

    grid = np.zeros((h, w), dtype=np.uint8)

    # Rasterize wall points
    for x, z in walls_xz:
        ix = int((x - xmin) / RESOLUTION)
        iz = int((z - zmin) / RESOLUTION)
        if 0 <= ix < w and 0 <= iz < h:
            grid[iz, ix] = 255

    # Save floor plan
    img = Image.fromarray(grid)
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    img.save("floorplan.png")

    print("Saved floor plan → floorplan.png")

# ============================================================
# MAIN
# ============================================================
def validate(flags):
    return os.path.exists(os.path.join(flags.path, "rgb.mp4"))

def main():
    flags = read_args()
    if not validate(flags):
        print("Not a valid StrayScanner dataset.")
        return

    generate_floor_plan(flags)

if __name__ == "__main__":
    main()
