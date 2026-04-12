import os
import numpy as np
from PIL import Image
from spot_semantic_mapping.spot.spot_ds import SpotDataset
import pandas as pd
from utils.jax_helper import cosine_similarity_jax, cdist
from tqdm import tqdm


def load_data(file_path):
    res = {
        "query_images": None,
        "db_images": None,
        "ground_truth": None,
        "ts": None
    }

    # Helper function to load and numerically sort images from a directory
    def load_images_from_directory(directory):
        if not os.path.exists(directory):
            print(f"Warning: Directory not found - {directory}")
            return None
            
        # Get all .jpg files
        file_names = [f for f in os.listdir(directory) if f.endswith('.jpg')]
        
        # Sort numerically by ID (e.g., 1.jpg, 2.jpg ... 10.jpg)
        file_names.sort(key=lambda x: int(os.path.splitext(x)[0]))
        
        images = []
        for file_name in file_names:
            img_path = os.path.join(directory, file_name)
            # Convert to RGB to ensure consistent 3-channel arrays
            img = Image.open(img_path).convert('RGB')
            images.append(np.array(img))
            
        return np.array(images)

    # 1. Load query images
    query_dir = os.path.join(file_path, "query")
    res["query_images"] = load_images_from_directory(query_dir)
    
    # 2. Load database (reference) images
    ref_dir = os.path.join(file_path, "ref")
    res["db_images"] = load_images_from_directory(ref_dir)

    # 3. Load ground truth
    gt_path = os.path.join(file_path, "my_ground_truth_new.npy")
    if os.path.exists(gt_path):
        res["ground_truth"] = np.load(gt_path, allow_pickle=True)
    else:
        print(f"Warning: Ground truth file not found - {gt_path}")

    res["ts"] = [i for i in range(len(res["query_images"]))]

    return res

def load_spot_data(spot_ds_path="dataset/spot/millerst/data", 
                   db_path="dataset/3578aa5730", 
                   cameras=[
                       "hand_color_image",
                       "frontright_fisheye_image", "frontleft_fisheye_image",
                       "left_fisheye_image", "right_fisheye_image",
                       "back_fisheye_image"
                   ], 
                   step=10,
                   window_size=3.0):
    
    res = {
        "ts": [],
        "query_images": [],
        "query_traj": None,
        "db_images": [],
        "db_traj": None,
        "ground_truth": None
    }

    # ==========================================
    # 1. Load Database Images
    # ==========================================
    db_frames_dir = os.path.join(db_path, "rgb_frames")
    db_odom_path = os.path.join(db_path, "odometry_spot_aligned.csv")
    db_coords = []

    if os.path.exists(db_odom_path) and os.path.exists(db_frames_dir):
        # Read odometry and sort chronologically just to be safe
        df_odom = pd.read_csv(db_odom_path)
        df_odom = df_odom.sort_values(by=' frame')
        
        # Map integer frame IDs to actual filenames (handles zero-padded names safely)
        db_filenames = [f for f in os.listdir(db_frames_dir) if f.endswith(('.jpg', '.png'))]
        frame_to_file = {int(os.path.splitext(f)[0]): f for f in db_filenames}
        
        # Single loop: Only load an image if it has a corresponding odometry row
        step_counter = 0
        for _, row in df_odom.iterrows():
            if step_counter % step != 0:
                step_counter += 1
                continue
            frame_id = int(row[' frame'])
            if frame_id in frame_to_file:
                # 1. Load Image
                fname = frame_to_file[frame_id]
                img_path = os.path.join(db_frames_dir, fname)
                img = Image.open(img_path).convert('RGB')
                res["db_images"].append(np.array(img))
                
                # 2. Load Coordinates
                db_coords.append([row[' x'], row[' y'], row[' z']])
            step_counter += 1
                
        # Convert to numpy arrays
        res["db_images"] = np.array(res["db_images"])
        db_coords = np.array(db_coords)
        
    else:
        if not os.path.exists(db_frames_dir):
            print(f"Warning: DB frames directory not found at {db_frames_dir}")
        if not os.path.exists(db_odom_path):
            print(f"Warning: DB odometry not found at {db_odom_path}")

    res["db_traj"] = db_coords

    # ==========================================
    # 3. Load Query Images & Odometry
    # ==========================================
    ds = SpotDataset(spot_ds_path) 
    query_coords = []

    for t in tqdm(range(len(ds))):
        snapshot = ds[t]
        
        # Extract and stack frames
        frames = [snapshot.cameras.get(cam).image for cam in cameras if snapshot.cameras.get(cam) is not None]
        frames = [frame for frame in frames if frame is not None] 

        if frames:
            max_H = max(frame.shape[0] for frame in frames)
            max_W = max(frame.shape[1] for frame in frames)
            frames_stacked = np.array([np.array(Image.fromarray(frame).resize((max_W, max_H))) for frame in frames])
            res["query_images"].extend(frames_stacked)
            for _ in frames:
                res["ts"].append(t)

        # --- SAFELY EXTRACT POSITION ---
        try:
            pos = snapshot.odom_T_body.position
            query_coords.append(pos) 
        except AttributeError:
            # If the sensor failure killed odometry too, append infinity 
            # so cdist doesn't crash and the array indices stay perfectly aligned.
            query_coords.append([np.inf, np.inf, np.inf])

    query_coords = np.array(query_coords)
    db_coords = np.array(db_coords)
    res["query_traj"] = query_coords
    res["query_images"] = np.array(res["query_images"])
    # ==========================================
    # 4. Compute Ground Truth Matrix
    # ==========================================
    if len(query_coords) > 0 and len(db_coords) > 0:
        distances = cdist(query_coords, db_coords)
        boolean_gt = distances <= window_size  
        
        gt_list = []
        for i, row in enumerate(boolean_gt):
            matching_db_indices = np.where(row)[0].tolist()
            gt_list.append([i, matching_db_indices])
            
        res["ground_truth"] = np.array(gt_list, dtype=object)
    else:
        print("Warning: Missing query or db coordinates.")    
    
    return res