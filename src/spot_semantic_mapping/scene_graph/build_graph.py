"""
End-to-End 3D Semantic Scene Graph Construction Pipeline (ConceptGraph implementation)
=========================================================

This module implements a full embodied perception pipeline that transforms
RGB-D video sequences into a structured 3D semantic scene graph.

The system integrates:

• Multi-frame RGB-D reconstruction
• YOLO + SAM instance segmentation
• 3D point cloud projection and clustering
• Multi-view object tracking and fusion
• Vision-language captioning (multi-view fusion)
• Text + vision embedding generation
• Object merging via geometric + semantic similarity
• Sparse VLM-based relation extraction
• Final scene graph + semantic point cloud export

Designed for embodied AI, robotic mapping, and world-model construction.

---------------------------------------------------------------------

Pipeline Overview
-----------------

1) Data Loading
   - RGB frames (video or folder)
   - Depth + confidence maps
   - Camera intrinsics
   - Odometry poses

2) Detection & Segmentation
   - YOLO detector for bounding boxes
   - SAM backend for mask refinement
   - Mask filtering & containment resolution

3) 3D Projection
   - Masked RGB-D → point cloud
   - DBSCAN spatial filtering
   - Transform into world frame
   - Build Detection objects

4) Multi-Frame Tracking
   - ConceptGraph-style object merging
   - Geometric + semantic similarity matching
   - Persistent object IDs

5) Multi-View Captioning
   - Select top-K informative crops
   - Caption each view
   - Fuse captions into a 3-word object name
   - Generate text embeddings

6) Object Pruning & Merging
   - Weighted similarity:
       w_sem, w_geo, w_text, w_clip
   - Distance cutoff + merge threshold

7) Relation Extraction
   - Spatial neighborhood pruning
   - VLM voting across top frames
   - Sparse semantic relation graph

8) Output
   - scene_graph.json
   - semantic colored .ply
   - tracker checkpoints

---------------------------------------------------------------------

Core Components
---------------

Detection Backend:
    YOLODetector (configurable)
    SAM (MobileSAM / SAM2 / FastSAM)

Vision Encoder:
    SigLIP for image embeddings

Captioner:
    LLM/VLM-based multi-view caption fusion

Tracker:
    ObjectTracker3D
    ConceptGraph-inspired object association

Relation Builder:
    build_sparse_scene_graph_edges(...)

---------------------------------------------------------------------

Key Design Principles
---------------------

• Open-set recognition via language supervision
• Multi-view object identity stabilization
• Geometry + semantics weighted merging
• Sparse relational reasoning (not full O(N²))
• Memory-efficient incremental processing
• Checkpoint-safe long sequences

---------------------------------------------------------------------

Coordinate Conventions
----------------------

• Depth projected using scaled intrinsics
• Point clouds transformed by T_WC
• All geometry stored in world frame (meters)
• Optional floor-only constraint

---------------------------------------------------------------------

Configuration
-------------

All hyperparameters are controlled via cfg:

cfg.camera
cfg.segmentation
cfg.tracking
cfg.pipeline
cfg.dbscan
cfg.logging
cfg.landmarks

---------------------------------------------------------------------

Entry Point
-----------

python pipeline.py <dataset_path> [--rotate] [--floor_only]
                   [--output_dir] [--scene_graph_only]
                   [--grayscale]

Returns:
    tracker (ObjectTracker3D) with:
        tracker.objects
        tracker.edges

---------------------------------------------------------------------

Research Context
----------------

This pipeline is designed for:

• Embodied scene graph construction
• Indoor world modeling
• Robotics + VLM reasoning
• Long-horizon memory compression
• 3D semantic knowledge base building

It provides the structural backbone for experiments on:
    - World-aware agents
    - Embodiment gap analysis
    - Scene graph optimal transport
    - Knowledge seed compression

---------------------------------------------------------------------

Dependencies
------------

• PyTorch
• Open3D
• OpenCV
• NumPy
• PIL
• tqdm
• Custom Model + utils modules

---------------------------------------------------------------------

Author Context
--------------

Built for large-scale RGB-D embodied scene understanding research.
"""


import os
import argparse

import numpy as np

np.float = np.float64
np.int = np.int_

from tqdm import tqdm
from PIL import Image
import open3d.t as o3d
import skvideo.io
import torch
import gc
import matplotlib.cm as cm

from configs.loader import cfg
from models.models import *
from spot_semantic_mapping.scene_graph.utils.graph_tracker import *
from spot_semantic_mapping.scene_graph.utils.relations_tracker import *

from utils.geometry import *
from utils.io import load_conf, load_intrinsics, load_poses, load_depth
from utils.mask import *
from utils.graph import *
from utils.logger import *
import os
from datetime import datetime

def save_graph(tracker, path="temp_graph.json"):
    assert path.endswith(".json"), "Graph path must end with .json"
    
    graph = {
        "nodes": [],
        "edges": []
    }
    for obj in tracker.objects:
        node = {
            "oid": obj.oid,
            "class_name": obj.class_name,
            "text_ft": obj.text_ft.cpu().numpy().tolist() if obj.text_ft is not None else None,
            "clip_ft": obj.clip_ft.cpu().numpy().tolist() if obj.clip_ft is not None else None,
            "position": obj.bbox.get_center().tolist() if obj.pcd is not None else None,
        }   
        graph["nodes"].append(node)
    for edge in tracker.edges:
        graph["edges"].append({
            "src_id": edge.src_id,
            "dst_id": edge.dst_id,
            "src_name": edge.src,
            "dst_name": edge.dst,
            "relation": edge.rtype
        })
        
    with open(path, "w") as f:
        import json
        json.dump(graph, f, indent=4)

def save_semantics_cloud(tracker, path="temp.ply"):
    unique_classes = list(set(obj.class_name for obj in tracker.objects))
    class_to_color = {cls: cm.get_cmap('tab20')(i / len(unique_classes))[:3] for i, cls in enumerate(unique_classes)}

    pc = o3d.geometry.PointCloud()
    for obj in tracker.objects:
        # Assign a color based on the object's class name
        color = class_to_color[obj.class_name]
        obj_colors = np.tile(color, (len(obj.pcd.points), 1))  # Repeat color for all points

        # Add points and their colors to the point cloud
        obj_pcd = obj.pcd
        obj_pcd.colors = o3d.utility.Vector3dVector(obj_colors)
        pc += obj_pcd

    o3d.io.write_point_cloud(path, pc)

def clean_gpu():
    torch.cuda.empty_cache()   # clears cached blocks
    gc.collect()               # Python GC
    print("[GPU] Cache cleared.")

def save_image_for_debug(img: np.ndarray, path: str):
    """
    Save an image (H, W, 3) uint8 for debugging.
    """
    img_pil = Image.fromarray(img)
    img_pil.save(path)

# =============================================================
# YOLO → SAM pipeline (robust, clean)
# =============================================================
def batch_embed_object_labels(
    tracker,
    embedder,
    batch_size: int = 128,
    normalize: bool = True,
    labels = [],
    obj_indices = []
):
    """
    Adds a text embedding to each object based on obj.class_name (or caption).

    Stores the embedding at obj.<attr_name> as a torch.float32 tensor.

    - Deduplicates identical labels to reduce API calls.
    - Batches requests for efficiency.
    """

    if len(labels) == 0:
        return

    label_to_objs = {}
    for name, i in zip(labels, obj_indices):
        label_to_objs.setdefault(name, []).append(i)

    unique_labels = list(label_to_objs.keys())

    label_to_vec = {}
    for start in tqdm(range(0, len(unique_labels), batch_size), desc="Embedding labels"):
        batch = unique_labels[start : start + batch_size]
        vecs = embedder(batch)  # (B, D) np.ndarray float32

        if normalize:
            norms = (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
            vecs = vecs / norms

        for name, v in zip(batch, vecs):
            label_to_vec[name] = v

    for name, idxs in label_to_objs.items():
        v = label_to_vec[name]
        t = torch.tensor(v, dtype=torch.float32, device=cfg.device)
        for i in idxs:
            setattr(tracker.objects[i], "text_ft", t)

def build_detection_and_sam_backends(cfg):
    """
    Build:
      - YOLODetector (bbox detector)
      - SAMMasker   (mask generator from bboxes)
    """

    det_name = cfg.segmentation.get("detector", "yolo").lower()

    if det_name in ["yolo", "yolo11", "yolov8"]:
        detection_model = YOLODetector(cfg)
    else:
        raise ValueError(f"Unknown detector backend: {det_name}")

    sam_name = cfg.segmentation.get("sam_backend", "mobilesam").lower()

    if sam_name == "mobilesam":
        sam_predictor = MobileSAMPredictor(cfg)
    elif sam_name == "sam2":
        sam_predictor = SAM2Predictor(cfg)
    elif sam_name == "fastsam":
        sam_predictor = FastSAMPredictor(cfg)
    else:
        raise ValueError(f"Unknown SAM backend: {sam_name}")

    return detection_model, sam_predictor

def run_yolo_sam(rgb, detection_model, sam_predictor, cfg):
    """
    INPUT:
        rgb  : np.ndarray (H, W, 3), uint8, rotated/upscaled already

    RETURNS:
        masks: list of (H, W) boolean arrays
        class_ids:  (N,)
        confidences: (N,)
        bboxes: (N, 4) float32
    """

    # =========================================================
    # Step 1 — YOLO DETECTION
    # =========================================================
    bboxes, class_ids, confs = detection_model(rgb)

    # No detection → return empty outputs
    if len(bboxes) == 0:
        return [], np.array([]), np.array([]), np.empty((0, 4), dtype=np.float32)

    # ultralytics SAM-compatible tensor format
    bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32, device=cfg.device)

    # =========================================================
    # Step 2 — SAM MASKING
    # =========================================================
    masks_np = sam_predictor(rgb, bboxes_tensor)

    if masks_np is None or len(masks_np) == 0:
        return [], class_ids, confs, bboxes
    if masks_np.ndim == 4:
        masks_np = masks_np[:, 0]

    masks = [m.astype(bool) for m in masks_np]

    return masks, class_ids, confs, bboxes

def mask_subtract_contained(xyxy: np.ndarray, mask: np.ndarray, th1=0.8, th2=0.7):
    """
    Remove nested masks:
    If mask_j is mostly inside mask_i, subtract j from i.
    """
    N = xyxy.shape[0]
    areas = (xyxy[:,2] - xyxy[:,0]) * (xyxy[:,3] - xyxy[:,1])

    lt = np.maximum(xyxy[:,None,:2], xyxy[None,:,:2])
    rb = np.minimum(xyxy[:,None,2:], xyxy[None,:,2:])
    inter = (rb - lt).clip(min=0)
    inter_area = inter[:,:,0] * inter[:,:,1]

    inter_over_box1 = inter_area / (areas[:,None] + 1e-6)
    inter_over_box2 = inter_area / (areas[None,:] + 1e-6)

    # j contained by i if:
    #  - j is mostly inside i (big overlap wrt j)
    #  - but j is not covering i
    contained = (inter_over_box1 < th2) & (inter_over_box2 > th1)

    # remove diagonal
    np.fill_diagonal(contained, False)

    mask_out = mask.copy()
    ii, jj = np.where(contained)

    # subtract mask_j from mask_i
    for i, j in zip(ii, jj):
        mask_out[i] &= (~mask_out[j])

    return mask_out

def filter_masks(masks, confs, detection_ids, cfg):
    """
    Filter AND merge masks while keeping confs + class_ids aligned.
    """

    # ============================================================
    # Step 1 — PRE-FILTER (size, confidence, structural)
    # ============================================================
    keep = []
    H = cfg.camera["depth_height"]
    W = cfg.camera["depth_width"]
    min_area = cfg.segmentation["min_mask_area_percent"]

    for i, m in enumerate(masks):
        area = np.sum(m) / (H * W)
        if area < min_area:
            continue
        if confs[i] < cfg.segmentation.get("min_confidence", 0.0):
            continue
        clsname = cfg.landmarks['classes'][detection_ids[i]]
        if clsname in ["floor", "ceiling"]:
            continue

        keep.append(i)

    masks = [masks[i] for i in keep]
    confs = [confs[i] for i in keep]
    detection_ids = [detection_ids[i] for i in keep]

    if len(masks) == 0:
        return [], [], []

    # ============================================================
    # Step 2 — RESOLVE CONTAINMENT MASKS
    # (subtract nested objects: mask_subtract_contained)
    # ============================================================
    # build fake xyxy from mask extents
    xyxy = []
    for m in masks:
        ys, xs = np.where(m)
        if len(xs) == 0:
            xyxy.append([0,0,0,0])
        else:
            xyxy.append([xs.min(), ys.min(), xs.max(), ys.max()])
    xyxy = np.array(xyxy, dtype=np.int32)

    masks_np = np.stack(masks).astype(bool)
    masks_np = mask_subtract_contained(xyxy, masks_np)
    masks = [masks_np[i] for i in range(len(masks_np))]

    # ============================================================
    # Step 3 — TRUNCATE IF TOO MANY MASKS
    # ============================================================
    max_masks = cfg.segmentation.get("max_masks", 60)
    if len(masks) > max_masks:
        masks = masks[:max_masks]
        confs = confs[:max_masks]
        detection_ids = detection_ids[:max_masks]

    # ============================================================
    # Step 4 — FINAL RESIZE STEP
    # ============================================================
    for i in range(len(masks)):
        if masks[i].shape != (H, W):
            mask = masks[i].astype(np.uint8) * 255
            mask_resized = cv2.resize(mask, (H, W), interpolation=cv2.INTER_NEAREST)
            masks[i] = (mask_resized > 0)

    return masks, detection_ids, confs

# =============================================================
# Object captioning
# =============================================================
def combine_crops(obj, max_views=5):
    """
    Select the top `max_views` crops from an object based on:
      1) area (height*width)
      2) expressiveness (RGB variance)
    Then pad them to equal size and horizontally concatenate.
    """

    if len(obj.crops) == 0:
        return None

    # ---------------------------------------------------
    # 1. Score crops
    # ---------------------------------------------------
    obj.sort_crops()

    # ---------------------------------------------------
    # 2. Select top-K crops
    # ---------------------------------------------------
    selected = [c for c, _, _, _ in obj.crops[:max_views]]
    
    # If fewer than K crops exist, continue with available ones
    if len(selected) == 0:
        return None

    # ---------------------------------------------------
    # 3. Pad to max height & width
    # ---------------------------------------------------
    max_h = max(c.shape[0] for c in selected)
    max_w = max(c.shape[1] for c in selected)

    padded = []
    for crop in selected:
        h, w = crop.shape[:2]

        pad_h = max_h - h
        pad_w = max_w - w

        padded_crop = np.pad(
            crop,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode='constant',
            constant_values=0
        )
        padded.append(padded_crop)

    return np.hstack(padded), selected

def caption_obj(obj, model, max_views=5):
    """
    Combine multi-view crops and produce a concise, accurate 3-word description.

    model(prompt, image=...)  must return a text caption.
    """

    # select best crops and combine them
    combined_strip, selected_crops = combine_crops(obj, max_views=max_views)

    if selected_crops is None or len(selected_crops) == 0:
        return "(no views)"

    # Caption each crop individually (strong prompt)
    crop_captions = []
    prompt_single = (
        "You are analyzing a single view of the same object. "
        "Describe the central object only, ignoring all background. "
        "Describe it in 3–5 words using only nouns and essential adjectives. "
        "Do NOT mention angles or camera views."
    )

    for crop in selected_crops:
        caption = model(prompt_single, image=crop)
        crop_captions.append(caption.strip())

    # Fuse into a final high-quality caption
    structured_list = "\n".join(f"- {c}" for c in crop_captions)

    prompt_fuse = (
        "You are given multiple captions of different views (the views are also included) of the SAME object in an indoor environment.\n"
        "Use the examples below to understand the required output format.\n\n"
        
        "Example 1:\n"
        "Input captions:\n"
        "- wooden table top\n"
        "- brown desk surface\n"
        "- wood grain panel\n"
        "Output (3 words):\n"
        "wooden table surface\n\n"
        
        "Example 2:\n"
        "Input captions:\n"
        "- black office chair\n"
        "- cushioned swivel seat\n"
        "- rolling desk chair\n"
        "Output (3 words):\n"
        "black office chair\n\n"

        "Now process the following captions from multiple views of ONE object:\n"
        f"{structured_list}\n\n"

        "Instructions:\n"
        "- Output must be EXACTLY 3 words.\n"
        "- Only output the final 3-word noun phrase.\n"
        "- No verbs, no explanations, no reasoning, no sentences.\n"
        "- Do not output anything except the 3-word answer.\n"
        "- If uncertain, choose the simplest and most visually grounded description.\n"
    )


    final_caption = model(prompt_fuse, image=combined_strip)
    final_caption = final_caption.strip()

    return final_caption

# ============================================================
#                    MAIN PIPELINE STRUCTURE
# ============================================================

def main(dataset_path, rotate, floor_only, cfg, ouput_dir=None, scene_graph_only=False, grayscale=False):
    # --------------------------------------------------------
    # 1. Load dataset (poses, intrinsics, depth, rgb frames)
    # --------------------------------------------------------
    # - load camera intrinsics from csv
    # - load per-frame odometry
    # - open RGB video reader
    # - list depth frames + confidence maps
    # --------------------------------------------------------
    DEPTH_WIDTH = cfg.camera["depth_width"]
    DEPTH_HEIGHT = cfg.camera["depth_height"]
    RGB_WIDTH = cfg.camera["rgb_width"]
    RGB_HEIGHT = cfg.camera["rgb_height"]
    if not ouput_dir:
        OUTPUT_DIR = cfg.pipeline.get("output_dir", "outputs")
    else:
        OUTPUT_DIR = ouput_dir
    EXP_NAME = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    intrinsics = load_intrinsics(
        os.path.join(dataset_path, "camera_matrix.csv"), 
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

    poses = load_poses(os.path.join(dataset_path, "odometry.csv"))
    depth_path = os.path.join(dataset_path, "depth")
    confidence_path = os.path.join(dataset_path, "confidence")
    rgb_path = os.path.join(dataset_path, "rgb.mp4")
    use_frames = False

    if not os.path.exists(rgb_path):
        if not os.path.exists(os.path.join(dataset_path, "rgb_frames")):
            raise FileNotFoundError(f"RGB video not found at {rgb_path}, and rgb_frames folder does not exist.")
        use_frames = True

    if not use_frames:
        # Extract and save all video frames
        video = skvideo.io.vreader(rgb_path)
    else:
        video = sorted(
            [os.path.join(dataset_path, "rgb_frames", f) for f in os.listdir(os.path.join(dataset_path, "rgb_frames")) if f.endswith(".png" )]
        )
        video = [np.array(Image.open(f).convert("RGB")) for f in video]
    
    if not scene_graph_only:
        logger = build_logger(EXP_NAME, log_dir=os.path.join(OUTPUT_DIR, EXP_NAME, cfg.logging.get("log_dir", "logs")))
    else:
        logger = None
        
    if floor_only:
        z_cap = min(p[2, 3] for p in poses) + 0.1 # only keep points near the floor, this way we avoid point captured standing on objects
    else:
        z_cap = np.inf
        
    # --------------------------------------------------------
    # 2. Initialize models (segmentation, vision encoder, captioner)
    # --------------------------------------------------------
    # - SegmentationModel (YOLO or FastSAM)
    # - SigLIP model (open-set image embedding + text embedding)
    # - Captioner (Llama-4-scout or BLIP)
    # --------------------------------------------------------
    
    print("\n[Init] Loading segmentation backend...")
    detection_model, sam_predictor = build_detection_and_sam_backends(cfg)
    vision_encoder = SiglipModel(cfg)
    captioner = GroqModel("meta-llama/llama-4-scout-17b-16e-instruct")
    print("[Init] Models loaded.\n")

    # --------------------------------------------------------
    # 3. Initialize trackers + buffers
    # --------------------------------------------------------
    if cfg.pipeline["resume_from_checkpoint"] is not None:
        # print(cfg.pipeline["resume_from_checkpoint"])
        tracker, resume_idx = load_full_tracker(cfg.pipeline["resume_from_checkpoint"], logger)
        print(f"[Resume] Resuming from checkpoint at frame {resume_idx}.")
    else:
        tracker = ObjectTracker3D(
            voxel_size=cfg.tracking["voxel_size"],
            w_geo=cfg.tracking["w_geo"],
            w_sem=cfg.tracking["w_sem"],
            match_threshold=cfg.tracking["match_threshold"],
        )
        resume_idx = 0
        print("[Resume] Starting from scratch.")
    
    # --------------------------------------------------------
    # 4. Iterate over frames, extract all objects
    # --------------------------------------------------------
    # For each frame:
    #   a. skip frames based on cfg.pipeline.use_every_n_frames
    #   b. load RGB, depth, confidence
    #   c. segmentation → masks
    #   d. merge overlapping masks
    #   e. extract crops + run SigLIP embeddings
    #   f. project masks to 3D
    # --------------------------------------------------------
    
    print("Processing frames and building point cloud...")
    frame_limit = cfg.pipeline.get("max_frames", len(poses))
    frame_count = 0
    total_frames = len(poses)
    if frame_limit == -1:
        frame_limit = len(poses)
    for idx, (T_WC, rgb) in enumerate(
        tqdm(zip(poses, video), desc="Processing frames")
    ):
        if floor_only:
            T_WC[2,3] = min(T_WC[2,3], z_cap)
        # if (T_WC[2,3] > z_cap):
        #     # skipping non floor points
        #     if floor_only:
        #         T_WC[2,]
        if idx < resume_idx: # This is very stupid, but I am tired
            continue
        if idx % cfg.pipeline["use_every_n_frames"] != 0:
            continue
        if idx >= frame_limit:
            break
        
        # logging
        if not scene_graph_only:
            if idx % cfg.logging["gpu_log_interval"] == 0:
                log_gpu_memory(logger, tag=f"Frame {idx}")
                
            if idx % cfg.logging.get("checkpoint_every", 1000) == 0:
                save_full_tracker(tracker, os.path.join(OUTPUT_DIR, EXP_NAME, "checkpoints"), idx, cfg, logger)
                save_semantics_cloud(tracker, path=os.path.join(OUTPUT_DIR, EXP_NAME, f"semantics_cloud_frame_{idx:06d}.ply"))

        # depth and confidence loading
        confidence = load_conf(os.path.join(confidence_path, f"{idx:06d}.png"))
        depth = load_depth(
            os.path.join(depth_path, f"{idx:06d}.png"),
            confidence,
            filter_level=cfg.projection["min_confidence"]
        )
        
        # resize rgb to depth size
        rgb = Image.fromarray(rgb)
        if grayscale:
            rgb = rgb.convert("L").convert("RGB")
        rgb_resized = rgb.resize((DEPTH_WIDTH, DEPTH_HEIGHT))
        
        # Ensure depth matches the expected dimensions
        if depth.shape[1] != DEPTH_WIDTH or depth.shape[0] != DEPTH_HEIGHT:
            depth = cv2.resize(depth, (DEPTH_WIDTH, DEPTH_HEIGHT), interpolation=cv2.INTER_NEAREST)

        rgb_downscaled = np.array(rgb.resize((
            int(RGB_WIDTH / cfg.segmentation["downscale_factor"] ),
            int(RGB_HEIGHT / cfg.segmentation["downscale_factor"] )
        )).rotate(rotate, expand=True))
        rgb_up = np.array(rgb_resized.rotate(rotate, expand=True))
        
        if not os.path.exists(os.path.join(dataset_path, "rgb_frames", f"{idx:06d}.png")):
            Image.fromarray(rgb_downscaled).save(os.path.join(dataset_path, "rgb_frames", f"{idx:06d}.png"))
        
        # perform detections + mask generation
        masks, class_ids, confidences, _ = run_yolo_sam(
            rgb_up, detection_model, sam_predictor, cfg
        )

        # filter masks
        masks, class_ids, confidences = filter_masks(masks, confidences, class_ids, cfg)
            
        detection_list = DetectionList()
        
        # 1) Prepare shapes
        H_up, W_up = rgb_up.shape[:2]
        H_down, W_down = rgb_downscaled.shape[:2]

        # 2) Compute scale factors
        sx = W_down / W_up
        sy = H_down / H_up

        for m, cid in zip(masks, class_ids):
            # bbox from rgb_up
            ys, xs = np.where(m > 0)
            x1_up, x2_up = xs.min(), xs.max()
            y1_up, y2_up = ys.min(), ys.max()

            # 3) Scale bbox into rgb_downscaled coordinates
            x1 = int(x1_up * sx)
            x2 = int(x2_up * sx)
            y1 = int(y1_up * sy)
            y2 = int(y2_up * sy)

            # 4) padding
            pad = cfg.segmentation["crop_padding"]
            x_b1 = max(0, x1 - pad)
            x_b2 = min(W_down, x2 + pad)
            y_b1 = max(0, y1 - pad)
            y_b2 = min(H_down, y2 + pad)

            # 5) high-quality crop
            crop = rgb_downscaled[y_b1:y_b2, x_b1:x_b2]

            if any(np.array(crop.shape[:2]) < pad):
                continue

            # embed crop using vision encoder
            feat = vision_encoder.embed_images([crop])[0]  # (D,)
            feat = feat / np.linalg.norm(feat)

            # create point cloud
            # rotate mask back to match depth
            m_rot = np.rot90(m, k=rotate//-90)
            
            depth_masked = depth.copy()
            depth_masked[m_rot == 0] = 0

            rgb_masked = np.array(rgb_resized).copy()
            rgb_masked[m_rot == 0] = 0
            rgbd_obj = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(rgb_masked),
                o3d.geometry.Image(depth_masked),
                depth_scale=1.0,
                depth_trunc=cfg.camera["max_depth"],
                convert_rgb_to_intensity=False
            )
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_obj, o3d_intrinsics
            )
            
            pcd = apply_dbscan(
                pcd,
                eps=cfg.dbscan["eps"],
                min_samples=cfg.dbscan["min_samples"]
            )

            # transform into world frame
            pcd.transform(T_WC)
            if len(pcd.points) == 0:
                continue
            
            # -----------------------------
            # Build detection object
            # -----------------------------
            det = Detection(
                pcd=pcd,
                bbox=pcd.get_axis_aligned_bounding_box(),
                clip_ft=torch.tensor(feat, dtype=torch.float32),
                # xyxy=np.array([x_b1, y_b1, x_b2, y_b2], dtype=np.int32),
                class_id=int(cid),
                class_name=detection_model.class_names[int(cid)],
                # frame_idx=idx,
            )
            det.set_crop((crop, np.array([x_b1, y_b1, x_b2, y_b2], dtype=np.int32), idx))

            detection_list.append(det)
        # --------------------------------------------
        # Update tracker using ConceptGraph pipeline
        # --------------------------------------------
        if len(detection_list) > 0:
            tracker.update(detection_list)
        frame_count += 1
    
    if not scene_graph_only:
        logger.info("Frame processing complete. Detected {}".format(len(tracker.objects)))
        logger.info("Processed {} frames out of {}".format(frame_count, total_frames))
        logger.info("Start captioning")
 
    # now tracker should store all objects, with merged geometry + sigclip features
    # Next steps
    # 1. run captioning per object based on multi-view crops
    # 2. create edge
           
    # this can be done through multi threading, but groq hates that
    object_names = []
    object_indices = []
    for object in tqdm(tracker.objects, desc="Captioning objects"):
        if object.class_name in cfg.landmarks['classes']:
            object.class_name = caption_obj(object, captioner, 5)
        if not object.oid:
            object.oid = tracker.objects.index(object)
        object_names.append(object.class_name)
        object_indices.append(object.oid)
    
    # generate semantic embedding feature
    text_embedder = OpenaiEmbedding("text-embedding-ada-002")
    batch_embed_object_labels(
        tracker,
        embedder=text_embedder,
        batch_size=128,     # tune down if you hit request limits
        normalize=True,
        labels = object_names,
        obj_indices=object_indices
    )
    
    # clean up and prune graphs
    # some criterias for pruning
    # - objects that are both geometrically close (via clustering) and semantically similar label, here we enforce high semantically similarity
    tracker = prune_and_merge_tracker_objects(
        tracker,
        w_sem=0.6,
        w_geo=0.4,
        w_text=0,
        w_clip=1,
        merge_threshold=0.6,
        geo_cutoff_m=0.4,
        logger=logger
    )
    if not scene_graph_only:
        save_full_tracker(tracker, os.path.join(OUTPUT_DIR, EXP_NAME, "checkpoints"), idx, cfg, logger)    
        logger.info("Captioning, Embedding, and Pruning complete.")    
        logger.info("Extracting semantic relations via VLM...")
    
    edges = []
    loader = lambda fidx: np.array(Image.open(os.path.join(dataset_path, "rgb_frames", f"{fidx:06d}.png")).convert("RGB"))
    edges = build_sparse_scene_graph_edges(
        tracker,
        frame_loader=loader,
        vlm_model=captioner,  # whatever you’re using for relation VLM
        logger=logger,
        neighbors_per_node=3,   # ~3 per node
        max_radius_m=2.0,       # tune based on room scale
        top_k_frames=3,         # VLM votes per pair (keep small)
        pad=12
    )
    tracker.edges = edges
    if not scene_graph_only:
        save_full_tracker(tracker, os.path.join(OUTPUT_DIR, EXP_NAME, "checkpoints"), idx, cfg, logger) 
        logger.info("Semantic relation extraction complete.")

        # --------------------------------------------------------
        # 5. Save final semantic point cloud + scene graph
        # --------------------------------------------------------
        save_semantics_cloud(tracker, path=os.path.join(OUTPUT_DIR, EXP_NAME, f"semantics_cloud_frame_{idx:06d}.ply"))
        
    if not scene_graph_only:
        save_graph(tracker, path=os.path.join(OUTPUT_DIR, EXP_NAME, f"scene_graph.json"))
        logger.info("Saved final semantic point cloud and scene graph.")
    else:
        save_graph(tracker, path=os.path.join(OUTPUT_DIR, f"scene_graph.json"))        

    return tracker
    

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("--rotate", type=int, default=-90, help="Rotate the input images by the specified degrees.")
    parser.add_argument("--floor_only", action="store_true", help="Only build floor plan based on points near the floor.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for results.")
    parser.add_argument("--scene_graph_only",  action="store_true", help="only store scene graph json at output dir")
    parser.add_argument("--grayscale", action="store_true", help="Use grayscale images for construction.")
    args = parser.parse_args()
    tracker = main(args.path, args.rotate, args.floor_only, cfg, args.output_dir, args.scene_graph_only, args.grayscale)