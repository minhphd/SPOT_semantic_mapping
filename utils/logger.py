import os
import sys
import time
import json
import logging
import numpy as np
import torch
from logging.handlers import RotatingFileHandler
from Model.tracker import Detection, ObjectTracker3D, MapObjectList, MapObject
import open3d as o3d
from Model.relations import RelationEdge
import shutil
import pickle

# ============================================================
# Colors
# ============================================================
class _Color:
    RESET   = "\033[0m"
    RED     = "\033[31m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    CYAN    = "\033[36m"


LEVEL_COLOR = {
    logging.INFO: _Color.GREEN,
    logging.WARNING: _Color.YELLOW,
    logging.ERROR: _Color.RED,
    logging.DEBUG: _Color.CYAN,
}


class ColorFormatter(logging.Formatter):
    def format(self, record):
        color = LEVEL_COLOR.get(record.levelno, _Color.RESET)
        record.msg = f"{color}{record.msg}{_Color.RESET}"
        return super().format(record)


def build_logger(
    name="main",
    log_dir="logs",
    filename="run.log",
    max_bytes=5 * 1024 * 1024,
    backup_count=5,
    level=logging.INFO,
):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, filename)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if len(logger.handlers) == 0:
        # Console
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(ColorFormatter("[%(asctime)s] [%(levelname)s] %(message)s",
                                       "%H:%M:%S"))
        logger.addHandler(ch)

        # Rotating file
        fh = RotatingFileHandler(log_path, maxBytes=max_bytes, backupCount=backup_count)
        fh.setFormatter(logging.Formatter(
            "[%(asctime)s] [%(levelname)s] (%(filename)s:%(lineno)d) - %(message)s"
        ))
        logger.addHandler(fh)

    return logger


# ============================================================
# GPU Memory Logger
# ============================================================
def log_gpu_memory(logger, tag="GPU"):
    if not torch.cuda.is_available():
        logger.info("CUDA not available.")
        return
    alloc = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)
    logger.info(f"{tag}: alloc={alloc:.1f}MB reserved={reserved:.1f}MB")


# ============================================================
# -------------------- CHECKPOINT --------------------
# ============================================================
def save_map_object(obj, path, max_crops=5):
    os.makedirs(path, exist_ok=True)

    # save points
    np.save(os.path.join(path, "points.npy"), np.asarray(obj.pcd.points))

    # save bbox (store min/max instead of 8 points)
    bbox = np.concatenate([obj.bbox.get_min_bound(), obj.bbox.get_max_bound()])
    np.save(os.path.join(path, "bbox.npy"), bbox)

    # save CLIP feature
    np.save(os.path.join(path, "clip_ft.npy"), obj.clip_ft.cpu().numpy())

    # ----------------------------
    # Save a limited number of crops
    # ----------------------------
    obj.sort_crops()
    for i, crop_tuple in enumerate(obj.crops[:max_crops]):
        with open(os.path.join(path, f"crop_{i}.pkl"), "wb") as f:
            pickle.dump(crop_tuple, f)

    meta = {
        "oid": obj.oid,
        "class_ids": obj.class_ids,
        "class_name": obj.class_name,
        "num_views": obj.num_views,
    }

    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

def save_full_tracker(tracker, out_dir, last_frame_idx, cfg, logger):
    """
    Save the entire tracker state:
      - all MapObjects
      - metadata (last processed frame, config snapshot)
      - geometric/semantic edges
    """
    os.makedirs(out_dir, exist_ok=True)
    obj_root = os.path.join(out_dir, "objects")

    # clean object directory
    if os.path.exists(obj_root):
        for filename in os.listdir(obj_root):
            file_path = os.path.join(obj_root, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    os.makedirs(obj_root, exist_ok=True)

    logger.info(f"[Checkpoint] Saving tracker with {len(tracker.objects)} objects.")

    # Save all objects
    for i, obj in enumerate(tracker.objects):
        path = os.path.join(obj_root, f"object_{i:04d}")
        save_map_object(obj, path)

    # ---------------------------------------------
    # Save metadata + edges
    # ---------------------------------------------
    edges_list = []
    if hasattr(tracker, "edges"):
        for e in tracker.edges:
            edges_list.append(e.__dict__)

    meta = {
        "last_frame_idx": last_frame_idx,
        "timestamp": time.time(),
        "num_objects": len(tracker.objects),
        "edges": edges_list,                        # <<< HERE
        "tracking_params": {
            "voxel_size": tracker.voxel_size,
            "w_geo": tracker.w_geo,
            "w_sem": tracker.w_sem,
            "match_threshold": tracker.match_threshold,
            "center_dist_thresh": getattr(tracker, "center_dist_thresh", None),
            "class_gate": getattr(tracker, "class_gate", False),
        },
        "cfg_snapshot": cfg.get_json(),
    }

    with open(os.path.join(out_dir, "tracker_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"[Checkpoint] Saved metadata → {out_dir}/tracker_meta.json")
    logger.info("[Checkpoint] Done.")


def load_full_tracker(checkpoint_dir, logger):
    """
    Rebuild full ObjectTracker3D from checkpoint.
    Returns:
        tracker  – restored tracker
        resume_idx – last completed frame index
    """

    logger.info(f"[Resume] Loading tracker checkpoint from: {checkpoint_dir}")

    # ---------------------------------------------
    # Load metadata
    # ---------------------------------------------
    meta_path = os.path.join(checkpoint_dir, "tracker_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing tracker_meta.json in {checkpoint_dir}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    resume_idx = meta["last_frame_idx"] + 1
    tracking_params = meta["tracking_params"]

    # ---------------------------------------------
    # Instantiate empty tracker
    # ---------------------------------------------
    tracker = ObjectTracker3D(
        voxel_size=tracking_params["voxel_size"],
        w_geo=tracking_params["w_geo"],
        w_sem=tracking_params["w_sem"],
        match_threshold=tracking_params["match_threshold"],
        center_dist_thresh=tracking_params["center_dist_thresh"],
        class_gate=tracking_params["class_gate"],
        edges=[],
    )

    # ---------------------------------------------
    # Load objects
    # ---------------------------------------------
    obj_root = os.path.join(checkpoint_dir, "objects")
    if not os.path.exists(obj_root):
        raise FileNotFoundError("Missing objects/ directory inside checkpoint.")

    objects = []
    for name in sorted(os.listdir(obj_root)):
        path = os.path.join(obj_root, name)
        try:
            obj = load_single_map_object(path)
            objects.append(obj)
            logger.info(f"[Resume] Loaded {name}")
        except Exception as e:
            logger.error(f"[Resume] Failed to load {name}: {e}")

    tracker.objects = MapObjectList(objects)
    logger.info(f"[Resume] Loaded {len(objects)} objects.")

    # ---------------------------------------------
    # Rebuild edges
    # ---------------------------------------------
    edge_dicts = meta.get("edges", [])
    edges = []
    for ed in edge_dicts:
        try:
            edge = RelationEdge(
                src_id=ed["src_id"],
                dist_id=ed["dist_id"], #typo in original code
                src=ed["src"],
                dst=ed["dst"],
                rtype=ed["rtype"],
                score=ed["score"],
                dist=ed["dist"],
            )
            edges.append(edge)
        except Exception as e:
            logger.error(f"[Resume] Failed to parse edge: {e}")

    tracker.edges = edges
    logger.info(f"[Resume] Loaded {len(edges)} edges.")

    logger.info(f"[Resume] Resume will start after frame {resume_idx}")

    return tracker, resume_idx


def load_single_map_object(path):
    # load geometry
    pts = np.load(os.path.join(path, "points.npy"))
    bbox_data = np.load(os.path.join(path, "bbox.npy"))
    feat = np.load(os.path.join(path, "clip_ft.npy"))

    # restore point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    # restore bbox (min/max)
    bbox = o3d.geometry.AxisAlignedBoundingBox(
        bbox_data[:3], bbox_data[3:6]
    )

    # load meta
    with open(os.path.join(path, "meta.json"), "r") as f:
        meta = json.load(f)

    # load crops
    crops = []
    for fname in sorted(os.listdir(path)):
        if fname.startswith("crop_") and fname.endswith(".pkl"):
            with open(os.path.join(path, fname), "rb") as f:
                crops.append(pickle.load(f))

    # reconstruct map object
    obj = MapObject(
        pcd=pcd,
        oid=meta.get("oid", None),
        bbox=bbox,
        clip_ft=torch.tensor(feat, dtype=torch.float32),
        crops=crops,
        class_ids=meta["class_ids"],
        class_name=meta["class_name"],
        detections=[],      # not restored — detections are ephemeral
        num_views=meta.get("num_views", len(crops)),
    )

    return obj

# ============================================================
# Timer Context
# ============================================================
class Timer:
    def __init__(self, logger, msg):
        self.logger = logger
        self.msg = msg

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, t, v, tb):
        dt = (time.time() - self.start) * 1000
        self.logger.info(f"{self.msg}: {dt:.2f} ms")
