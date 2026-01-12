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
from Model.models import *
from Model.tracker import *

from Model.relations import *
from utils.geometry import *
from utils.io import load_conf, load_intrinsics, load_poses, load_depth
from utils.mask import *
from utils.graph import *
from utils.logger import *
import os
from datetime import datetime


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

def main(dataset_path, cfg):
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
    OUTPUT_DIR = cfg.pipeline.get("output_dir", "outputs")
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
    
    if not os.path.exists(rgb_path):
        raise FileNotFoundError(f"Cannot find: {rgb_path}")
    
    # Extract and save all video frames
    video = skvideo.io.vreader(rgb_path)
    
    logger = build_logger(EXP_NAME, log_dir=os.path.join(OUTPUT_DIR, EXP_NAME, cfg.logging.get("log_dir", "logs")))

    # --------------------------------------------------------
    # 2. Initialize models (segmentation, vision encoder, captioner)
    # --------------------------------------------------------
    # - SegmentationModel (YOLO or FastSAM)
    # - SigLIP model (open-set image embedding + text embedding)
    # - BLIP captioner (multi-view)
    # --------------------------------------------------------
    
    print("\n[Init] Loading segmentation backend...")
    detection_model, sam_predictor = build_detection_and_sam_backends(cfg)
    vision_encoder = SiglipModel(cfg)
    captioner = GroqModel("meta-llama/llama-4-scout-17b-16e-instruct")
    print("[Init] Models loaded.\n")

    # --------------------------------------------------------
    # 3. Initialize trackers + buffers
    # --------------------------------------------------------
    if cfg.pipeline["resume_from_checkpoint"]:
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

    if frame_limit == -1:
        frame_limit = len(poses)
    for idx, (T_WC, rgb) in enumerate(
        tqdm(zip(poses, video), desc="Processing frames")
    ):
        if idx < resume_idx: # This is very stupid, but I am tired
            continue
        if idx % cfg.pipeline["use_every_n_frames"] != 0:
            continue
        if idx >= frame_limit:
            break
        
        # logging
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
        rgb_resized = rgb.resize((DEPTH_WIDTH, DEPTH_HEIGHT))
        rgb_downscaled = np.array(rgb.resize((
            int(RGB_WIDTH / cfg.segmentation["downscale_factor"] ),
            int(RGB_HEIGHT / cfg.segmentation["downscale_factor"] )
        )).rotate(-90, expand=True))
        rgb_up = np.array(rgb_resized.rotate(-90, expand=True))
        
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
            m_rot = np.rot90(m, k=1)
            
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
    
    
    # topdown = tsdf_extract_topdown(tsdf_canvas, res=0.01, up_axis="y")
    # Image.fromarray(topdown).save("topdown_tsdf.png")

    # save_image_for_debug(
    #     topdown,
    #     path=os.path.join(OUTPUT_DIR, EXP_NAME, "top_down_projection.png")
    # )
    logger.info("Frame processing complete. Detected {}".format(len(tracker.objects)))
    logger.info("Start captioning")
 
    # now tracker should store all objects, with merged geometry + sigclip features
    # Next steps
    # 1. run captioning per object based on multi-view crops
    # 2. create edge
    # 3. Hardest part: room and places formulation + (maybe parsing a floor plan to a vision model?)
       
    # this can be done through multi threading, but groq hates that
    for object in tqdm(tracker.objects, desc="Captioning objects"):
        if object.class_name in cfg.landmarks['classes']:
            object.class_name = caption_obj(object, captioner, 5)
        if not object.oid:
            object.oid = tracker.objects.index(object)
    save_full_tracker(tracker, os.path.join(OUTPUT_DIR, EXP_NAME, "checkpoints"), idx, cfg, logger)    
    logger.info("Captioning complete.")
    
    edges = []
    logger.info("Extracting semantic relations via VLM...")
    loader = lambda fidx: np.array(Image.open(os.path.join(dataset_path, "rgb_frames", f"{fidx:06d}.png")).convert("RGB"))
    llm_model = OpenaiModel('gpt-5-mini')
    edges = compute_vlm_relations(tracker.objects, loader, captioner, llm_model)
    tracker.edges = edges
    save_full_tracker(tracker, os.path.join(OUTPUT_DIR, EXP_NAME, "checkpoints"), idx, cfg, logger) 
    tracker.edges = edges
    logger.info("Semantic relation extraction complete.")

    logger.info("Now we perform room detection")
    # we will generate floor plan first based on the global point cloud
    # then detect rooms based on connected components
    # then assign objects to rooms based on their centroids
    
    
    save_semantics_cloud(tracker, path=os.path.join(OUTPUT_DIR, EXP_NAME, f"semantics_cloud_frame_{idx:06d}.ply"))
    

    return tracker
    

def build_flat_cloud_and_feats(tracker):
    """
    Flatten all per-object point clouds into a single point cloud,
    and build:
      - feats: (N_obj, D) normalized SigLIP feats
      - offsets: list of (start_idx, end_idx) for each object in flat cloud
      - names: class_name list (len N_obj)
    """
    all_points = []
    all_colors = []
    offsets = []
    feats = []
    names = []

    cur = 0
    for obj in tracker.objects:
        if obj.pcd is None or obj.clip_ft is None:
            continue

        pts = np.asarray(obj.pcd.points)
        if pts.shape[0] == 0:
            continue

        # flatten geometry
        all_points.append(pts)

        # initial neutral gray color
        base_color = np.array([0.4, 0.4, 0.4], dtype=np.float32)
        all_colors.append(np.tile(base_color, (pts.shape[0], 1)))

        start = cur
        end = cur + pts.shape[0]
        offsets.append((start, end))
        cur = end

        # store normalized feature + class name
        ft = obj.clip_ft.detach().cpu().numpy().astype(np.float32)
        ft /= (np.linalg.norm(ft) + 1e-6)

        feats.append(ft)
        names.append(obj.class_name)

    if len(all_points) == 0:
        raise RuntimeError("No objects with points + embeddings in tracker.")

    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)
    feats = np.vstack(feats)

    flat_pcd = o3d.geometry.PointCloud()
    flat_pcd.points = o3d.utility.Vector3dVector(all_points)
    flat_pcd.colors = o3d.utility.Vector3dVector(all_colors)

    return flat_pcd, feats, offsets, names

def interactive_glow_loop(tracker, vision_encoder, out_dir="outputs"):
    """
    Terminal REPL: type a text prompt, we recompute similarity for each object
    and dump a recolored point cloud that "glows" where semantics match.

    View `outputs/semantic_glow.ply` on your laptop and reload after each prompt.
    """

    os.makedirs(out_dir, exist_ok=True)

    print("\n[Glow] Precomputing flattened cloud + object embeddings...")
    flat_pcd, obj_feats, offsets, names = build_flat_cloud_and_feats(tracker)
    base_points = np.asarray(flat_pcd.points)

    print(f"[Glow] {len(names)} objects, {base_points.shape[0]} points total.")
    print("[Glow] Enter prompts.")

    while True:
        try:
            prompt = input("\nPrompt> ").strip()
            
            # 1) Encode text
            with torch.no_grad():
                txt_ft = vision_encoder.embed_texts([prompt])[0]
                txt_ft = torch.tensor(txt_ft, dtype=torch.float32)
                txt_ft = txt_ft / (torch.norm(txt_ft) + 1e-6)
                txt_ft = txt_ft.detach().cpu().numpy().astype(np.float32)

            # 2) Similarity per object
            sims = obj_feats @ txt_ft  # cosine (features are normalized)
            # normalize to [0,1] for color scaling
            sims_norm = (sims - sims.min()) / (sims.max() - sims.min() + 1e-6)

            print("[Glow] Similarity per class:")
            for name, s in zip(names, sims):
                print(f"  {name:20s}: {s:.3f}")

            # 3) Build colors for all points
            new_colors = np.zeros((base_points.shape[0], 3), dtype=np.float32)

            cmap = cm.get_cmap("inferno")

            for i, (start, end) in enumerate(offsets):
                sim_n = sims_norm[i]

                # bright color for high similarity; darker for low
                col = np.array(cmap(sim_n)[:3], dtype=np.float32)
                col = col * (0.3 + 0.7 * sim_n)  # extra "glow" for high sim

                new_colors[start:end, :] = col

            flat_pcd.colors = o3d.utility.Vector3dVector(new_colors)

            # 4) Save PLY
            safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in prompt)
            out_path = os.path.join(out_dir, "semantic_glow.ply")
            o3d.io.write_point_cloud(out_path, flat_pcd)

            print(f"[Glow] Wrote recolored cloud for '{prompt}' → {out_path}")
            print("       Open/reload this file in your local viewer to see the glow.")
        except KeyboardInterrupt:
            print("\n[Glow] Exiting due to keyboard interrupt.")
            break


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    args = parser.parse_args()
    tracker = main(args.path, cfg)

    # 2) Build SigLIP encoder (text + image)
    print("\n[Init] Loading SigLIP encoder for interactive glow...")
    vision_encoder = SiglipModel(cfg)
    print("[Init] SigLIP ready.")

    # 3) Interactive prompt loop
    # interactive_glow_loop(tracker, vision_encoder)

