import os
import numpy as np
import open3d as o3d
from PIL import Image
np.float = np.float64
np.int = np.int_
import skvideo.io
from scipy.spatial.transform import Rotation
import argparse

DEPTH_WIDTH = 256
DEPTH_HEIGHT = 192
MAX_DEPTH = 20.0

def load_intrinsics(path):
    cam = np.loadtxt(path, delimiter=",")
    sx = DEPTH_WIDTH / 1920
    sy = DEPTH_HEIGHT / 1440

    return o3d.camera.PinholeCameraIntrinsic(
        width=DEPTH_WIDTH,
        height=DEPTH_HEIGHT,
        fx=cam[0,0] * sx,
        fy=cam[1,1] * sy,
        cx=cam[0,2] * sx,
        cy=cam[1,2] * sy
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

def main(dataset_path, conf_level):

    # -----------------------
    # Load metadata
    # -----------------------
    intr = load_intrinsics(os.path.join(dataset_path, "camera_matrix.csv"))
    poses = load_poses(os.path.join(dataset_path, "odometry.csv"))

    depth_dir = os.path.join(dataset_path, "depth")
    conf_dir = os.path.join(dataset_path, "confidence")
    depth_fns = sorted([f for f in os.listdir(depth_dir) if f.endswith(".npy") or f.endswith(".png")])

    video = skvideo.io.vreader(os.path.join(dataset_path, "rgb.mp4"))

    # -----------------------
    # TSDF volume
    # -----------------------
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.02,
        sdf_trunc=0.05,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    # -----------------------
    # Integrate all frames
    # -----------------------
    for i, (T_WC, rgb_frame) in enumerate(zip(poses, video)):
        if i == 0:
            continue  # Skip first frame (often incomplete)
        
        if i >= len(depth_fns):
            break

        print(f"Integrating {i:06}", end="\r")

        # Load depth + confidence filtering
        depth_mm = load_depth(os.path.join(depth_dir, depth_fns[i]))
        conf = load_conf(os.path.join(conf_dir, f"{i:06}.png"))

        # Apply confidence mask
        depth_mm[conf < conf_level] = 0.0

        depth_img = o3d.geometry.Image(depth_mm)
        
        # Resize RGB
        rgb = Image.fromarray(rgb_frame).resize((DEPTH_WIDTH, DEPTH_HEIGHT))
        rgb_np = np.array(rgb)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb_np),
            depth_img,
            depth_scale=1.0,
            depth_trunc=MAX_DEPTH,
            convert_rgb_to_intensity=False
        )

        T_CW = np.linalg.inv(T_WC)
        volume.integrate(rgbd, intr, T_CW)

    # -----------------------
    # Extract mesh
    # -----------------------
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    mesh_path = os.path.join(dataset_path, "hybrid_mesh.ply")
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    print(f"\nSaved mesh → {mesh_path}")

    # -----------------------
    # Sample clean point cloud
    # -----------------------
    pcd = mesh.sample_points_poisson_disk(400000)
    pcd_path = os.path.join(dataset_path, "hybrid_pointcloud.ply")
    o3d.io.write_point_cloud(pcd_path, pcd)
    print(f"Saved point cloud → {pcd_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("--confidence", "-c", type=int, default=1, help="0, 1, or 2")
    args = parser.parse_args()

    main(args.path, args.confidence)
