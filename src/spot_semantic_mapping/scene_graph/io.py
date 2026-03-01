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
                dst_id=ed["dst_id"], #typo in original code
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
