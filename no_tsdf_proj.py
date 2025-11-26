import os
import numpy as np
import open3d as o3d
from PIL import Image
np.float = np.float64
np.int = np.int_
import skvideo.io
from scipy.spatial.transform import Rotation
import argparse
import multiprocessing as mp

DEPTH_WIDTH = 256
DEPTH_HEIGHT = 192
MAX_DEPTH = 20.0


def load_intrinsics(path):
    cam = np.loadtxt(path, delimiter=",")
    sx = DEPTH_WIDTH / 1920
    sy = DEPTH_HEIGHT / 1440
    return (
        cam[0, 0] * sx,   # fx
        cam[1, 1] * sy,   # fy
        cam[0, 2] * sx,   # cx
        cam[1, 2] * sy    # cy
    )


def load_poses(path):
    odo = np.loadtxt(path, delimiter=",", skiprows=1)
    poses = []
    for row in odo:
        pos = row[2:5]
        quat = row[5:]
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(quat).as_matrix()
        T[:3, 3] = pos
        poses.append(T)
    return poses


def load_depth(path):
    if path.endswith(".npy"):
        depth_mm = np.load(path)
    else:
        depth_mm = np.array(Image.open(path))
    return depth_mm.astype(np.float32) / 1000.0


def load_conf(path):
    return np.array(Image.open(path))


# ------------------------------
# WORKER FUNCTION
# ------------------------------
def process_frame(
    i, pose, rgb_frame, depth_path, conf_path,
    fx, fy, cx, cy, conf_level
):

    # Build intrinsics locally (safe)
    intr = o3d.camera.PinholeCameraIntrinsic(
        DEPTH_WIDTH, DEPTH_HEIGHT, fx, fy, cx, cy
    )

    # Load depth + conf
    depth = load_depth(depth_path)
    conf = load_conf(conf_path)
    depth[conf < conf_level] = 0.0

    depth_img = o3d.geometry.Image(depth)

    # RGB
    rgb = Image.fromarray(rgb_frame).resize((DEPTH_WIDTH, DEPTH_HEIGHT))
    rgb_np = np.array(rgb)
    color_img = o3d.geometry.Image(rgb_np)

    # Build RGBD
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_img, depth_img,
        depth_scale=1.0,
        depth_trunc=MAX_DEPTH,
        convert_rgb_to_intensity=False
    )

    # Project to 3D
    T_CW = np.linalg.inv(pose)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr, extrinsic=T_CW)

    # Extract np arrays (picklable)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    print(f"Processed frame {i} ({len(points)} points)")
    return points, colors



def main(dataset_path, conf_level, every):

    fx, fy, cx, cy = load_intrinsics(os.path.join(dataset_path, "camera_matrix.csv"))
    poses = load_poses(os.path.join(dataset_path, "odometry.csv"))

    depth_dir = os.path.join(dataset_path, "depth")
    conf_dir = os.path.join(dataset_path, "confidence")

    depth_fns = sorted([
        f for f in os.listdir(depth_dir)
        if f.endswith(".npy") or f.endswith(".png")
    ])

    rgb_path = os.path.join(dataset_path, "rgb.mp4")
    video = skvideo.io.vreader(rgb_path)

    pool = mp.Pool(mp.cpu_count())
    jobs = []

    for i, (pose, depth_fn, rgb_frame) in enumerate(zip(poses, depth_fns, video)):
        if i == 0:
            continue
        if i % every != 0:
            continue

        depth_path = os.path.join(depth_dir, depth_fn)
        conf_path = os.path.join(conf_dir, f"{i:06}.png")

        jobs.append(pool.apply_async(
            process_frame,
            (
                i, pose, rgb_frame, depth_path, conf_path,
                fx, fy, cx, cy, conf_level
            )
        ))

    pool.close()
    pool.join()

    # Gather results
    all_points = []
    all_colors = []

    for job in jobs:
        pts, cols = job.get()
        all_points.append(pts)
        all_colors.append(cols)

    # Concatenate
    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)

    # Build PCD
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)

    # Save
    out_path = os.path.join(dataset_path, "raw_pointcloud_parallel.ply")
    o3d.io.write_point_cloud(out_path, pcd)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("--confidence", "-c", type=int, default=1)
    parser.add_argument("--every", "-e", type=int, default=1)
    args = parser.parse_args()

    main(args.path, args.confidence, args.every)
