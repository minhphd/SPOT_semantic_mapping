import pickle as pkl
import os
import open3d as o3d
import numpy as np

np.int = np.int_
np.float = np.float64


from utils.logger import load_full_tracker, build_logger
from scipy.spatial.transform import Rotation
from argparse import ArgumentParser
from PIL import Image
import skvideo.io
from tqdm import tqdm
from utils.io import load_intrinsics, load_poses, load_depth, load_conf
from configs.loader import cfg
import json
import cv2
import sys
import re

# How large the floor plan grid is
RESOLUTION = 0.05   # meters per pixel (5cm)
NORMAL_UP = np.array([0, 1, 0])   # Y-up for Stray Scanner

def parse_svg_path_to_xy(path_str: str) -> np.ndarray:
    """
    Parse a minimal SVG path like:
      Mx,y Lx,y Lx,y ... Z
    into an (N,2) float32 array of (x,y) points.

    Here, your 'path' coordinates look like (x, y) in world units.
    We'll interpret:
      x -> world X
      y -> world Z   (since your floorplan uses XZ)
    """
    # Extract all number pairs "x,y"
    pairs = re.findall(r"(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)", path_str)
    if not pairs:
        return np.zeros((0, 2), dtype=np.float32)
    pts = np.array([(float(a), float(b)) for a, b in pairs], dtype=np.float32)
    return pts

def world_xz_to_grid_ij(x: float, z: float, xmin: float, zmin: float, resolution: float):
    """
    Convert world (x,z) to grid indices (i,j) = (row=z, col=x).
    """
    j = int((x - xmin) / resolution)
    i = int((z - zmin) / resolution)
    return i, j

def polygon_world_to_grid(poly_xz: np.ndarray, xmin: float, zmin: float, resolution: float) -> np.ndarray:
    """
    poly_xz: (N,2) in world coords (x,z)
    returns: (N,2) int32 in image coords (col=x, row=z) for cv2
    """
    if poly_xz.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int32)

    cols = ((poly_xz[:, 0] - xmin) / resolution).astype(np.int32)
    rows = ((poly_xz[:, 1] - zmin) / resolution).astype(np.int32)
    pts = np.stack([cols, rows], axis=1)  # cv2 wants (x=col, y=row)
    return pts

def load_rooms_polygons(rooms_json_path: str):
    """
    Returns list of dicts: [{"room": name, "poly_xz": (N,2) world coords}, ...]
    """
    with open(rooms_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rooms = []
    for r in data.get("rooms", []):
        name = r.get("room", "")
        path = r.get("path", "")
        pts = parse_svg_path_to_xy(path)  # (x,y)
        # interpret y as z
        poly_xz = np.stack([pts[:, 0], pts[:, 1]], axis=1).astype(np.float32)
        rooms.append({"room": name, "poly_xz": poly_xz})
    return rooms

# ============================================================
# CLI
# ============================================================
description = "Generate a 2D floor plan from StrayScanner dataset"
def read_args():
    parser = ArgumentParser(description=description)    
    parser.add_argument('path', type=str, help="Path to the StrayScanner dataset")
    parser.add_argument('--tracker_path', type=str, default=None, help="Path to the tracker file (optional)")
    parser.add_argument('--save_img', action='store_true', help="Save the generated floor plan as an image")
    parser.add_argument('--every', type=int, default=10)
    parser.add_argument('--save_embeddings', action='store_true', help="Save the dense embedding field as a .npy file")
    parser.add_argument('--output_path', type=str, help="Path to save the output floor plan (.npy file)")
    
    # RANSAC / wall params (tweak if needed)
    parser.add_argument('--plane_dist_thresh', type=float, default=0.05) # How far points can be from plane to be inlier
    parser.add_argument('--plane_min_points', type=int, default=800) # minimum inlier points to accept a plane
    parser.add_argument('--max_planes', type=int, default=100)  # maximum number of planes to extract
    return parser.parse_args()

# ============================================================
# LOAD DATA
# ============================================================
def read_data(flags):
    logger = build_logger()
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
        "depth_frames": depth_frames,
        "tracker": load_full_tracker(flags.tracker_path, logger)[0] if flags.tracker_path else None,
        "logger": logger
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

        _, inliers = remaining.segment_plane(
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
    return grid, xmin, zmin

def plot_detected_objects(grid, tracker, xmin, zmin):
    """
    This function will plot detected objects from the tracker onto the occupancy grid.
    inputs:
        grid: 2D numpy array representing the occupancy grid
        tracker: tracker object containing detected objects and their positions
         - tracker.objects: MapObjectList containing detected objects
         - MapObject
            - MapObject.pcd: o3d.geometry.PointCloud of the object
            - pcd: o3d.geometry.PointCloud
                bbox: o3d.geometry.AxisAlignedBoundingBox
                clip_ft: torch.Tensor                   # aggregated feature
                crops: List[tuple] = field(default_factory=list)  # list of (crop, xyxy, frame_idx, score)
                oid: Optional[int] = None
                class_ids: List[int] = field(default_factory=list)
                class_name: str = ""
    outputs:
        grid: 2D numpy array with detected objects marked
    """
    embedding_field = np.zeros((grid.shape[0], grid.shape[1], tracker.objects[0].clip_ft.shape[-1]), dtype=np.float32)
    if tracker is None or not hasattr(tracker, "objects"):
        print("No tracker data available or invalid tracker format.")
        return grid

    for obj in tracker.objects:
        # Extract all points from the object's point cloud
        points = np.asarray(obj.pcd.points)

        # Project the points to the XZ plane
        points_xz = points[:, [0, 2]]

        # Convert to grid coordinates
        for x, z in points_xz:
            ix = int((x - xmin) / RESOLUTION)
            iz = int((z - zmin) / RESOLUTION)

            # Clip to grid boundaries
            if 0 <= ix < grid.shape[1] and 0 <= iz < grid.shape[0]:
                grid[iz, ix] = 128  # Mark with a gray value
                embedding_field[iz, ix] = obj.clip_ft.cpu().numpy()
                
    is_wall = (grid == 255)
    embedding_field[is_wall] = 0.0
    
    return grid, embedding_field

def add_room_att(tracker, graph_path):
    with open(graph_path, 'r') as f:
        graph = json.load(f)
        
    for obj in tracker.objects:
        oid = obj.oid
        room = graph['nodes'][oid]['room']
        setattr(obj, 'room', room)
    return tracker

from collections import deque

def fill_embedding_field_by_room(
    grid: np.ndarray,
    tracker,
    xmin: float,
    zmin: float,
    rooms_json_path: str,
    wall_value: int = 255,
    resolution: float = RESOLUTION,
    agg: str = "mean",
    default_mode: str = "zeros",   # "zeros" | "global_mean"
):
    """
    Create a dense embedding field using room prototypes.

    Each room cell gets the average embedding of all objects assigned to that room
    (tracker.objects must have obj.room and obj.clip_ft).
    """
    H, W = grid.shape
    assert tracker is not None and hasattr(tracker, "objects"), "tracker.objects required"
    D = int(tracker.objects[0].clip_ft.shape[-1])

    free = (grid != wall_value)

    # --- collect per-room object embeddings ---
    room_to_embs = {}
    all_embs = []

    for obj in tracker.objects:
        if not hasattr(obj, "clip_ft"):
            continue
        e = obj.clip_ft
        if hasattr(e, "detach"):
            e = e.detach().cpu().numpy()
        e = np.asarray(e).astype(np.float32).reshape(-1)

        if e.shape[0] != D:
            continue

        all_embs.append(e)

        room_name = getattr(obj, "room", None)
        if room_name is None:
            continue
        room_to_embs.setdefault(room_name, []).append(e)

    global_mean = None
    if len(all_embs) > 0:
        global_mean = np.mean(np.stack(all_embs, axis=0), axis=0).astype(np.float32)

    # --- compute room prototypes ---
    room_proto = {}
    for room_name, embs in room_to_embs.items():
        X = np.stack(embs, axis=0)
        if agg == "mean":
            proto = X.mean(axis=0)
        elif agg == "max":
            proto = X.max(axis=0)
        else:
            raise ValueError(f"Unknown agg='{agg}'")
        room_proto[room_name] = proto.astype(np.float32)

    # --- load polygons and paint masks ---
    rooms = load_rooms_polygons(rooms_json_path)

    room_id_map = -np.ones((H, W), dtype=np.int32)
    proto_list = []  # index -> embedding
    name_to_id = {}

    for ridx, r in enumerate(rooms):
        name = r["room"]
        poly_xz = r["poly_xz"]
        poly_ij = polygon_world_to_grid(poly_xz, xmin, zmin, resolution)  # (N,2) in (col,row)
        if poly_ij.shape[0] < 3:
            continue

        # clip polygon points into image bounds (optional)
        poly_ij[:, 0] = np.clip(poly_ij[:, 0], 0, W - 1)
        poly_ij[:, 1] = np.clip(poly_ij[:, 1], 0, H - 1)

        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [poly_ij], 1)

        # store id
        name_to_id[name] = len(proto_list)

        # choose prototype for this room
        if name in room_proto:
            proto = room_proto[name]
        else:
            if default_mode == "global_mean" and global_mean is not None:
                proto = global_mean
            else:
                proto = np.zeros((D,), dtype=np.float32)

        proto_list.append(proto)
        room_id_map[mask.astype(bool)] = name_to_id[name]

    proto_arr = np.stack(proto_list, axis=0).astype(np.float32) if len(proto_list) else None

    # --- build dense embedding field ---
    field = np.zeros((H, W, D), dtype=np.float32)

    if proto_arr is not None:
        valid = (room_id_map >= 0) & free
        field[valid] = proto_arr[room_id_map[valid]]

    # keep walls = 0
    field[~free] = 0.0

    return field, room_id_map, room_proto, name_to_id

def extrapolate_embeddings(
    embedding_field: np.ndarray,
    grid: np.ndarray,
    method: str = "laplace",
    wall_value: int = 255,
    seed_eps: float = 1e-8,
    # laplace params
    n_iters: int = 400,
    alpha: float = 1.0,
    tol: float = 1e-5,
    # nearest params
    use_8_conn: bool = False,
) -> np.ndarray:
    """
    Extrapolate sparse embeddings to a dense embedding field over free space.

    Args:
        embedding_field: (H,W,D) float32. Sparse: nonzero vectors at seed cells.
        grid: (H,W) uint8 occupancy. wall_value indicates obstacles.
        method: 'laplace' or 'nearest'
        wall_value: occupancy value marking walls/obstacles.
        seed_eps: threshold to consider a cell as a seed (known embedding).
        n_iters: number of iterations for laplace diffusion.
        alpha: laplace update strength. alpha=1.0 -> full Jacobi replacement per iter.
        tol: early stopping threshold on max change.
        use_8_conn: for 'nearest', use 8-connected BFS instead of 4-connected.

    Returns:
        dense_field: (H,W,D) float32. Walls remain zeros. Seeds preserved.
    """
    H, W, D = embedding_field.shape
    dense = embedding_field.astype(np.float32, copy=True)

    free = (grid != wall_value)
    seed_mask = (np.linalg.norm(dense, axis=-1) > seed_eps) & free

    # Always keep walls at 0
    dense[~free] = 0.0

    if method.lower() in ("nearest", "nn", "voronoi"):
        # Multi-source BFS assign each free cell embedding of nearest seed (geodesic)
        nearest_id = -np.ones((H, W), dtype=np.int32)
        dist = np.full((H, W), 1e9, dtype=np.int32)

        # store seed embeddings in a list to index by id
        seed_pos = np.argwhere(seed_mask)  # (N,2) in (i,j)
        if seed_pos.shape[0] == 0:
            # nothing to extrapolate from
            return dense

        seed_embs = dense[seed_mask]  # (N,D)

        q = deque()
        for nid, (i, j) in enumerate(seed_pos):
            nearest_id[i, j] = nid
            dist[i, j] = 0
            q.append((i, j))

        nbrs4 = [(-1,0),(1,0),(0,-1),(0,1)]
        nbrs8 = nbrs4 + [(-1,-1),(-1,1),(1,-1),(1,1)]
        nbrs = nbrs8 if use_8_conn else nbrs4

        while q:
            i, j = q.popleft()
            for di, dj in nbrs:
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W and free[ni, nj]:
                    nd = dist[i, j] + 1
                    if nd < dist[ni, nj]:
                        dist[ni, nj] = nd
                        nearest_id[ni, nj] = nearest_id[i, j]
                        q.append((ni, nj))

        out = np.zeros_like(dense, dtype=np.float32)
        valid = (nearest_id >= 0) & free
        out[valid] = seed_embs[nearest_id[valid]]
        # keep walls zeros
        out[~free] = 0.0
        # seeds preserved
        out[seed_mask] = dense[seed_mask]
        return out
    
    # if method.lower() in ("weighted_distance"):
    #     # cell embedding = distance weighted average of all embeddings in graph
        

    if method.lower() not in ("laplace", "harmonic", "diffuse"):
        raise ValueError(f"Unknown method='{method}'. Use 'nearest' or 'laplace'.")

    # ---------------------------
    # Laplacian harmonic extension (Jacobi iterations)
    # Constrained seeds: keep them fixed.
    # Only update free & non-seed cells.
    # ---------------------------

    # Precompute neighbor offsets (4-connected for stability)
    nbrs = [(-1,0),(1,0),(0,-1),(0,1)]

    # Initialize non-seed free cells with something reasonable:
    # If currently zero, you can optionally initialize with nearest fill for faster convergence.
    # (Good trick.) We'll do it cheaply:
    if np.any(free & ~seed_mask & (np.linalg.norm(dense, axis=-1) <= seed_eps)):
        dense = extrapolate_embeddings(dense, grid, method="nearest", wall_value=wall_value, seed_eps=seed_eps)

    # Now diffuse, keeping seeds clamped
    clamp = dense.copy()  # seed values
    update_mask = free & ~seed_mask

    for it in range(n_iters):
        prev = dense.copy()

        # Jacobi: new(x) = average of neighbors
        # We'll do it with explicit loops to keep it dependency-free.
        for i in range(H):
            for j in range(W):
                if not update_mask[i, j]:
                    continue

                acc = np.zeros((D,), dtype=np.float32)
                cnt = 0
                for di, dj in nbrs:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < H and 0 <= nj < W and free[ni, nj]:
                        acc += prev[ni, nj]
                        cnt += 1

                if cnt > 0:
                    newv = acc / cnt
                    # alpha blending (alpha=1.0 full replacement)
                    dense[i, j] = (1.0 - alpha) * prev[i, j] + alpha * newv

        # clamp seeds and walls
        dense[seed_mask] = clamp[seed_mask]
        dense[~free] = 0.0

        # early stopping
        max_delta = float(np.max(np.abs(dense - prev)))
        if max_delta < tol:
            # print(f"[laplace] converged at iter {it}, max_delta={max_delta:.2e}")
            break

    return dense

def main():
    flags = read_args()
    data = read_data(flags)
    grid, xmin, zmin = generate_floor_plan(flags)
    grid, embedding_field = plot_detected_objects(grid, data["tracker"], xmin, zmin)
    dense_field = extrapolate_embeddings(
        embedding_field,
        grid,
        method="nearest",
        use_8_conn=True
    )
    tracker = add_room_att(data["tracker"], "graph_dataset_grayscale/graph.json")
    coarse_field, room_id_map, room_proto, name_2_id = fill_embedding_field_by_room(
        grid,
        tracker,
        xmin,
        zmin,
        rooms_json_path="graph_dataset/rooms.json",
        wall_value=255,
        resolution=RESOLUTION,
        agg="mean",
        default_mode="zeros"
    )
    
    if flags.save_img:
        img = Image.fromarray(grid)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img.save(os.path.join(flags.output_path, "floorplan_with_objects.png"))
        print("Saved floor plan with detected objects → floorplan_with_objects.png")
    if flags.save_embeddings:
        np.save(os.path.join(flags.output_path, "dense_embedding_field.npy"), dense_field)
        np.save(os.path.join(flags.output_path, "occupancy_grid.npy"), grid)
        print("Saved dense embedding field → dense_embedding_field.npy")
        coarse_dict = {
            "coarse_field": coarse_field,
            "room_id_map": room_id_map,
            "room_proto": room_proto,
            "name_to_id": name_2_id,
            "id_to_name": {v:k for k,v in name_2_id.items()}
        }
        with open(os.path.join(flags.output_path, "coarse_embedding_field.pkl"), "wb") as f:
            pkl.dump(coarse_dict, f)
        print("Saved coarse embedding field → coarse_embedding_field.pkl")
if __name__ == "__main__":
    main()
