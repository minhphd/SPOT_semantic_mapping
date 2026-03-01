import os
import numpy as np
from PIL import Image
from spot.data_loading import SpotDataset
import pandas as pd
from scipy.spatial.distance import cdist
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class VPRImageDataset(Dataset):
    def __init__(self, image_data, is_spot_query=False, spot_ds_path=None, target_size=(480, 640)):
        """
        image_data: Either a list of file paths (DB images) 
                    or a list of dicts {"t": int, "cam": str} (Spot queries).
        """
        self.image_data = image_data
        self.is_spot_query = is_spot_query
        self.target_size = target_size
        
        # We initialize SpotDataset here so it's ready for lazy extraction
        self.spot_ds = SpotDataset(spot_ds_path) if is_spot_query and spot_ds_path else None

        # Optional: Add PyTorch transforms here (e.g., normalization for your model)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        item = self.image_data[idx]
        
        if self.is_spot_query:
            # Lazy load from Spot dataset
            t = item["t"]
            cam_name = item["cam"]
            
            img_array = self.spot_ds[t].cameras.get(cam_name).image
            img = Image.fromarray(img_array).convert('RGB')
        else:
            # Lazy load from disk path
            img = Image.open(item).convert('RGB')
            
        # Standardize sizes (since you were doing max_W, max_H previously, 
        # it's better to force a standard size for batching)
        img = img.resize(self.target_size)
        
        # Convert to tensor (or numpy array if you prefer)
        img_tensor = self.transform(img)
        
        return img_tensor
        

def _get_image_paths(directory):
    if not os.path.exists(directory):
        print(f"Warning: Directory not found - {directory}")
        return []
    file_names = [f for f in os.listdir(directory) if f.endswith('.jpg')]
    file_names.sort(key=lambda x: int(os.path.splitext(x)[0]))
    return [os.path.join(directory, f) for f in file_names]


def load_17places_data(file_path):
    res = {
        "query_images": None, # Will now hold PATHS
        "db_images": None,    # Will now hold PATHS
        "ground_truth": None,
        "ts": None
    }

    res["query_images"] = _get_image_paths(os.path.join(file_path, "query"))
    res["db_images"] = _get_image_paths(os.path.join(file_path, "ref"))

    gt_path = os.path.join(file_path, "my_ground_truth_new.npy")
    if os.path.exists(gt_path):
        res["ground_truth"] = np.load(gt_path, allow_pickle=True)
        
    res["ts"] = list(range(len(res["query_images"])))
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
        "query_images": [], # Will now hold tuples: (timestamp_t, camera_name)
        "query_traj": [],
        "db_images": [],    # Will now hold PATHS
        "db_traj": [],
        "ground_truth": None,
        "spot_ds_path": spot_ds_path # Save path to initialize lazy loader later
    }

    # ==========================================
    # 1. Load Database Image PATHS
    # ==========================================
    db_frames_dir = os.path.join(db_path, "rgb_frames")
    db_odom_path = os.path.join(db_path, "odometry_spot_aligned.csv")

    if os.path.exists(db_odom_path) and os.path.exists(db_frames_dir):
        df_odom = pd.read_csv(db_odom_path).sort_values(by=' frame')
        db_filenames = [f for f in os.listdir(db_frames_dir) if f.endswith(('.jpg', '.png'))]
        frame_to_file = {int(os.path.splitext(f)[0]): f for f in db_filenames}
        
        step_counter = 0
        for _, row in df_odom.iterrows():
            if step_counter % step != 0:
                step_counter += 1
                continue
            
            frame_id = int(row[' frame'])
            if frame_id in frame_to_file:
                # Store PATH instead of opening the PIL image
                img_path = os.path.join(db_frames_dir, frame_to_file[frame_id])
                res["db_images"].append(img_path)
                res["db_traj"].append([row[' x'], row[' y'], row[' z']])
                
            step_counter += 1
            
        res["db_traj"] = np.array(res["db_traj"])

    # ==========================================
    # 2. Load Query Metadata
    # ==========================================
    ds = SpotDataset(spot_ds_path) 
    
    for t in tqdm(range(len(ds)), desc="Scanning query metadata"):
        snapshot = ds[t]
        
        # Check which cameras are actually present for this timestamp
        valid_cameras = [cam for cam in cameras if snapshot.cameras.get(cam) is not None]
        
        if valid_cameras:
            for cam in valid_cameras:
                # Store instructions on how to load this image later
                res["query_images"].append({"t": t, "cam": cam})
                res["ts"].append(t)
        
        try:
            pos = snapshot.odom_T_body.position
            res["query_traj"].append([pos.x, pos.y, pos.z] if hasattr(pos, 'x') else pos)
        except AttributeError:
            res["query_traj"].append([np.inf, np.inf, np.inf])

    res["query_traj"] = np.array(res["query_traj"])

    # ==========================================
    # 3. Compute Ground Truth Matrix
    # ==========================================
    if len(res["query_traj"]) > 0 and len(res["db_traj"]) > 0:
        distances = cdist(res["query_traj"], res["db_traj"], metric='euclidean')
        boolean_gt = distances <= window_size  
        
        gt_list = [[i, np.where(row)[0].tolist()] for i, row in enumerate(boolean_gt)]
        res["ground_truth"] = np.array(gt_list, dtype=object)
    
    return res