"""
Scene Graph Checkpoint Loading & Tracker Restoration
====================================================

This module restores a full 3D semantic scene graph tracker from a saved
checkpoint directory. It reconstructs:

• ObjectTracker3D state
• All persistent MapObject instances
• Geometric point clouds
• Axis-aligned bounding boxes
• CLIP feature embeddings
• Multi-view crop history
• Relation edges
• Resume frame index for continued processing

Designed for long-running embodied scene graph pipelines where incremental
checkpointing and crash recovery are required.

---------------------------------------------------------------------

Checkpoint Directory Structure
-------------------------------

<checkpoint_dir>/
├── tracker_meta.json
├── objects/
│   ├── obj_0001/
│   │   ├── points.npy
│   │   ├── bbox.npy
│   │   ├── clip_ft.npy
│   │   ├── meta.json
│   │   ├── crop_000.pkl
│   │   └── ...
│   └── obj_0002/
│       └── ...
└── ...

tracker_meta.json contains:
    - last_frame_idx
    - tracking_params
    - serialized edges

---------------------------------------------------------------------

What Gets Restored
------------------

Objects:
    • 3D geometry (Open3D point cloud)
    • Bounding box
    • CLIP embedding
    • Multi-view crop memory
    • Object ID
    • Class metadata

Edges:
    • Source / destination IDs
    • Relation type
    • Score
    • Spatial distance

Tracker:
    • Matching thresholds
    • Weight parameters (w_geo, w_sem, etc.)
    • Center distance threshold
    • Class gating configuration

---------------------------------------------------------------------

Important Notes
---------------

• Detection history is NOT restored (detections are ephemeral).
• Objects are restored as MapObject instances.
• Resume index is returned to continue frame processing safely.
• This module assumes compatibility with ObjectTracker3D schema.

---------------------------------------------------------------------

Primary Functions
-----------------

load_full_tracker(checkpoint_dir, logger)
    Rebuild complete tracker and return:
        (tracker, resume_frame_idx)

load_single_map_object(path)
    Reconstruct a MapObject from saved geometry + metadata.

load_scene_graph(graph_path)
    Load exported JSON scene graph.

---------------------------------------------------------------------

Use Case
--------

• Crash-safe long sequences
• Large indoor reconstruction pipelines
• Resume training/inference
• Multi-hour robotic mapping sessions
• Experiment reproducibility

---------------------------------------------------------------------

Dependencies
------------

• Open3D
• NumPy
• PyTorch
• JSON
• pickle
• spot_semantic_mapping modules

---------------------------------------------------------------------

Research Context
----------------

This checkpoint mechanism enables scalable, incremental construction of
3D semantic world models without requiring full reprocessing from scratch.
It is essential for embodied AI systems operating over long time horizons.
"""

import os
import json
import open3d as o3d
import numpy as np
import torch
import pickle

from spot_semantic_mapping.scene_graph.utils.graph_tracker import Detection, ObjectTracker3D, MapObjectList, MapObject
from spot_semantic_mapping.scene_graph.utils.relations_tracker import RelationEdge


def load_scene_graph(graph_path):
    with open(graph_path, 'rb') as fp:
        graph = json.load(fp)

    return graph


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
