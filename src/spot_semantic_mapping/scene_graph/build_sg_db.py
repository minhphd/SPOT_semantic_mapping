import shutil
import numpy as np
import pandas as pd
from build_map import main as build_map_main
from configs.loader import cfg
import os

def load_odometry_csv(path: str) -> pd.DataFrame:
    """
    Load odometry CSV file into a DataFrame.
    Expects columns: timestamp, x, y, z, qx, qy, qz, qw
    """
    df = pd.read_csv(path)
    expected_cols = {"timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"}
    if not expected_cols.issubset(df.columns):
        missing = expected_cols - set(df.columns)
        raise ValueError(f"Missing columns in odometry CSV: {missing}")
    return df

def _cumulative_path_length(xyz: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(xyz[1:] - xyz[:-1], axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])

def sample_odometry_window_by_length(
    odo: pd.DataFrame,
    window_m: float,
    seed: int | None = 0,
    xyz_cols: tuple[str, str, str] = ("x", "y", "z"),
) -> pd.DataFrame:
    """
    Random contiguous window by path length (meters).
    Picks a random start index and finds the end index where traveled distance >= window_m.
    """
    if window_m <= 0:
        raise ValueError("window_m must be > 0")
    for c in xyz_cols:
        if c not in odo.columns:
            raise ValueError(f"Missing column '{c}' in odometry")

    xyz = odo.loc[:, list(xyz_cols)].to_numpy(dtype=np.float32)
    if len(xyz) < 2:
        raise ValueError("odometry must have at least 2 rows")

    cum = _cumulative_path_length(xyz)
    total = float(cum[-1])
    if total < window_m:
        raise ValueError(f"trajectory too short: total_length={total:.3f}m < window_m={window_m:.3f}m")

    rng = np.random.default_rng(seed)

    # choose a start such that we can reach window_m ahead
    # condition: cum[start] <= total - window_m
    max_start = np.searchsorted(cum, total - window_m, side="right") - 1
    max_start = max(0, int(max_start))
    start = int(rng.integers(0, max_start + 1))

    end_dist = cum[start] + window_m
    end = int(np.searchsorted(cum, end_dist, side="left"))
    end = min(end, len(odo) - 1)

    out = odo.iloc[start:end + 1].copy().reset_index(drop=True)
    return out

def construct_out(main_path, output_path, N_samples=1000, window_m=2.0, seed=67):
    """
    Build a semantic graph database from odometry data.
    Args:
        main_path (str): Path to the directory containing odometry.csv
        output_path (str): Path to save the semantic graph database CSV
        N_samples (int): Number of samples to generate per scene
    """
    os.makedirs(output_path, exist_ok=True)
    for room in os.listdir(main_path):
        room_path = os.path.join(main_path, room)
        if not os.path.isdir(room_path):
            continue
        for height in os.listdir(room_path):
            df = load_odometry_csv(os.path.join(room_path, height, 'odometry.csv'))
            
            # note that in each of these directories, file format is {frame:06d}.png
            confidence_path = os.path.join(room_path, height, 'confidence')
            depth_path = os.path.join(room_path, height, 'depth')
            rgb_path = os.path.join(room_path, height, 'rgb_frames')
            os.makedirs(os.path.join(output_path, room, height), exist_ok=True)
            for i in range(N_samples):
                os.makedirs(os.path.join(output_path, room, height, f"sample_{i:04d}"), exist_ok=True)
                local_path = os.path.join(output_path, room, height, f"sample_{i:04d}")
                
                subodo = sample_odometry_window_by_length(df, window_m=window_m, seed=seed + i)
                start_frame = int(subodo.iloc[0]['frame'])
                end_frame = int(subodo.iloc[-1]['frame'])
                
                os.makedirs(os.path.join(local_path, 'confidence'), exist_ok=True)
                os.makedirs(os.path.join(local_path, 'depth'), exist_ok=True)
                os.makedirs(os.path.join(local_path, 'rgb_frames'), exist_ok=True)
                for f in range(start_frame, end_frame + 1):
                    conf_src = os.path.join(confidence_path, f"{f-start_frame:06d}.png")
                    depth_src = os.path.join(depth_path, f"{f-start_frame:06d}.png")
                    rgb_src = os.path.join(rgb_path, f"{f-start_frame:06d}.png")
                    
                    conf_dst = os.path.join(local_path, 'confidence', f"{f-start_frame:06d}.png")
                    depth_dst = os.path.join(local_path, 'depth', f"{f-start_frame:06d}.png")
                    rgb_dst = os.path.join(local_path, 'rgb_frames', f"{f-start_frame:06d}.png")
                    
                    shutil.copyfile(conf_src, conf_dst)
                    shutil.copyfile(depth_src, depth_dst)
                    shutil.copyfile(rgb_src, rgb_dst)
                shutil.copyfile(os.path.join(room_path, height, 'camera_matrix.csv'), os.path.join(local_path, 'camera_matrix.csv'))
                subodo['frame'] = subodo['frame'] - start_frame
                subodo.to_csv(os.path.join(local_path, 'odometry.csv'), index=False)
        print(f"Built semantic subgraphs database for room: {room}")
    print(f"Semantic graph database built at {output_path}")
    
def construct_scene_graph_db(main_path: str, output_path: str, N_samples: int = 1000, window_m: float = 2.0, seed: int = 67):
    """
    Wrapper function to construct the scene graph database.
    """
    # construct_out(main_path, output_path, N_samples, window_m, seed)
    for room in os.listdir(output_path):
        for height in os.listdir(os.path.join(output_path, room)):
            for sample in os.listdir(os.path.join(output_path, room, height)):
                sample_path = os.path.join(output_path, room, height, sample)
                if not os.path.exists(os.path.join(sample_path, 'scene_graph.json')):
                    build_map_main(sample_path, 0, False, cfg, sample_path, scene_graph_only=True)
                    print(f"Built scene graph for sample: {sample_path}")
                else:
                    print(f"Scene graph already exists for sample: {sample_path}")
                
                
if __name__ == "__main__":
    construct_scene_graph_db(
        main_path='replica_rgbd',
        output_path='replica_sg_db',
        N_samples=50,
        window_m=3.0,
        seed=67
    )