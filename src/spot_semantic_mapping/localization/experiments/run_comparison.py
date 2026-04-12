"""
run_comparison.py
=================
Paper experiment: compare 11 localization methods across three families on
the SPOT indoor dataset.

Families & methods
------------------
graph2graph
  - ot_gw          : Gromov-Wasserstein matching between YOLO query-graph and DB subgraph
  - bag_of_objects  : YOLO crop SigLIP embeddings vs node clip_ft  (Sinkhorn OT)
  - bag_of_texts    : YOLO class-name text embeddings vs node text_ft (Sinkhorn OT)

im2graph
  - clip_avg        : SigLIP image CLS vs average of nearby DB node clip_ft
  - clip_max        : SigLIP image CLS vs max-similarity nearby DB node clip_ft
  - dino_graph      : DINO CLS vs nearby DB node clip_ft (cross-modal baseline)

im2im (→ position via image retrieval)
  - dino_cls        : DINOv2 CLS cosine retrieval
  - clip_cls        : SigLIP CLS cosine retrieval
  - dino_vlad       : DinoVLAD (AnyLoc) — patch VLAD
  - clip_vlad       : ClipVLAD — SigLIP patch VLAD
  - snap_loc        : DinoVLAD + graph re-scoring (SnapLoc)

Metrics
-------
- 3m accuracy  : % timesteps where predicted pos is within 3 m of GT
- 5m accuracy  : % timesteps where predicted pos is within 5 m of GT
- room accuracy: % timesteps where top-1 DB frame's room matches GT room
- mean / median position error (m)

Usage
-----
python run_comparison.py \\
  --dataset   .../spot_dataset_w_gt_5m.pkl \\
  --graph     .../scene_graph.json \\
  --mapping   ./mapping.txt \\
  [--floorplan  ./floorplan.png] \\
  [--yolo_cache .../yolo_cache.pkl] \\
  [--methods  all] \\
  [--top_k    10] \\
  [--graph_window 3.0] \\
  [--snap_alpha   0.4] \\
  [--output_dir   results/]
"""

from __future__ import annotations

import argparse
import os
import pickle as pkl
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ── project imports ──────────────────────────────────────────────────────────
from models.models import DinoModel, SiglipModel, YOLODetector
from configs.loader import cfg

from spot_semantic_mapping.localization.methods.VPR_im2im.img_encoder import ImageEncoder
from spot_semantic_mapping.localization.methods.VPR_im2im.localization import (
    prepare_embeddings,
    localize_at_t,
)
from spot_semantic_mapping.scene_graph.io import load_scene_graph
from utils.jax_helper import cosine_similarity_jax

from spot_semantic_mapping.localization.experiments.exp_metrics import (
    predict_position,
    compute_loc_metrics,
    aggregate_results,
    format_results_table,
)
from spot_semantic_mapping.localization.experiments.viz import (
    plot_heatmap,
    save_results_csv,
)

try:
    import ot as pot
except ImportError:
    pot = None

# ── constants ─────────────────────────────────────────────────────────────────
ALL_METHODS = [
    "dino_cls", "clip_cls", "dino_vlad", "clip_vlad", "snap_loc",
    "clip_avg", "clip_max", "dino_graph",
    "bag_of_texts", "bag_of_objects", "ot_gw",
]

GRAPH2GRAPH_METHODS = {"bag_of_texts", "bag_of_objects", "ot_gw"}

IM2IM_DINO_METHODS  = {"dino_cls", "dino_vlad", "snap_loc", "dino_graph"}
IM2IM_CLIP_METHODS  = {"clip_cls", "clip_vlad", "clip_avg", "clip_max",
                        "bag_of_texts", "bag_of_objects", "ot_gw"}


# ═══════════════════════════════════════════════════════════════════════════════
# Pre-computation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def build_db_node_index(db_traj: np.ndarray, graph: dict, window_m: float = 3.0) -> list[dict]:
    """
    For each DB frame, collect nearby scene-graph nodes (within window_m in XY).

    Returns a list of length N_db. Each entry is a dict:
        {"clip_ft": (K, D) | None, "text_ft": (K, D_t) | None, "positions": (K, 3)}
    Empty dicts indicate no nodes nearby.
    """
    nodes = graph.get("nodes", [])
    if len(nodes) == 0:
        print("[WARN] Scene graph has no nodes — im2graph / graph2graph will be zero scores")
        return [{"clip_ft": None, "text_ft": None, "positions": np.zeros((0, 3))}
                for _ in range(len(db_traj))]

    node_pos = np.array([n["position"] for n in nodes], dtype=np.float32)  # (M, 3)
    clip_fts  = [n.get("clip_ft") for n in nodes]
    text_fts  = [n.get("text_ft") for n in nodes]

    result = []
    for i, pos in enumerate(db_traj):
        dists = np.linalg.norm(node_pos[:, :2] - pos[:2], axis=1)  # 2D XY distance
        mask  = dists <= window_m

        nearby_clip = None
        nearby_text = None
        nearby_pos  = node_pos[mask]

        valid_clip = [np.asarray(clip_fts[j], dtype=np.float32)
                      for j in np.where(mask)[0] if clip_fts[j] is not None]
        valid_text = [np.asarray(text_fts[j], dtype=np.float32)
                      for j in np.where(mask)[0] if text_fts[j] is not None]

        if valid_clip:
            nearby_clip = _l2norm(np.stack(valid_clip, axis=0))
        if valid_text:
            nearby_text = _l2norm(np.stack(valid_text, axis=0))

        result.append({"clip_ft": nearby_clip, "text_ft": nearby_text,
                        "positions": nearby_pos})

    return result


def precompute_db_clip_aggregates(db_node_index: list[dict]) -> tuple[np.ndarray, list]:
    """
    Returns:
        db_clip_avg : (N_db, D) — mean clip_ft per DB frame (zeros where no nodes)
        db_clip_stack: list[np.ndarray | None] — raw (K,D) per DB frame for CLIP-Max
    """
    first_clip = next((e["clip_ft"] for e in db_node_index if e["clip_ft"] is not None), None)
    D = first_clip.shape[-1] if first_clip is not None else 1

    db_clip_avg   = np.zeros((len(db_node_index), D), dtype=np.float32)
    db_clip_stack = []

    for i, entry in enumerate(db_node_index):
        cf = entry["clip_ft"]
        if cf is not None and len(cf) > 0:
            db_clip_avg[i] = cf.mean(axis=0)
            db_clip_stack.append(cf)
        else:
            db_clip_stack.append(None)

    db_clip_avg = _l2norm(db_clip_avg)
    return db_clip_avg, db_clip_stack


def precompute_db_text_stack(db_node_index: list[dict]) -> list:
    """Returns list[np.ndarray | None] — raw (K, D_text) text embeddings per DB frame."""
    return [e["text_ft"] for e in db_node_index]


# ── YOLO cache ────────────────────────────────────────────────────────────────

def load_or_build_yolo_cache(
    dataset: dict,
    yolo_model,
    siglip_model,
    cache_path: str,
) -> dict | None:
    """
    Build (or load from cache) YOLO detections for every query timestep.

    Cache format: {t: [{"class_name": str, "clip_ft": (D,), "bbox_norm": (4,)}, ...]}

    Returns None if yolo_model is None.
    """
    if yolo_model is None:
        return None

    if cache_path and Path(cache_path).exists():
        print(f"[YOLO cache] Loading from {cache_path}")
        with open(cache_path, "rb") as f:
            return pkl.load(f)

    print("[YOLO cache] Building YOLO detection cache — this may take a while...")
    query_images = dataset["query_images"]
    ts           = np.array(dataset["ts"])
    T            = int(ts[-1]) + 1

    cache: dict[int, list] = defaultdict(list)

    for t in tqdm(range(T), desc="YOLO cache"):
        frames_at_t = query_images[ts == t]
        if len(frames_at_t) == 0:
            continue

        detections_at_t = []
        for frame in frames_at_t:
            bboxes, class_ids, confs = yolo_model(frame)
            if len(bboxes) == 0:
                continue

            H, W = frame.shape[:2]
            for bbox, cid, conf in zip(bboxes, class_ids, confs):
                x1, y1, x2, y2 = bbox.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame[y1:y2, x1:x2]
                clip_emb = siglip_model.embed_images([crop])
                if clip_emb is None:
                    continue

                class_name = yolo_model.class_names[cid] if cid < len(yolo_model.class_names) else "object"
                detections_at_t.append({
                    "class_name": class_name,
                    "clip_ft":    clip_emb[0].astype(np.float32),
                    "bbox_norm":  np.array([x1/W, y1/H, x2/W, y2/H], dtype=np.float32),
                    "conf":       float(conf),
                })

        cache[t] = detections_at_t

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pkl.dump(dict(cache), f)
        print(f"[YOLO cache] Saved → {cache_path}")

    return dict(cache)


# ═══════════════════════════════════════════════════════════════════════════════
# Score functions
# ═══════════════════════════════════════════════════════════════════════════════

def _score_im2graph_avg(query_emb: np.ndarray, db_clip_avg: np.ndarray) -> np.ndarray:
    """query_emb: (D,); db_clip_avg: (N_db, D) → (N_db,) scores."""
    q = _l2norm(query_emb[None])[0]
    return db_clip_avg @ q


def _score_im2graph_max(query_emb: np.ndarray, db_clip_stack: list) -> np.ndarray:
    """Max cosine similarity of query_emb against all nearby node embeddings per DB frame."""
    q = _l2norm(query_emb[None])[0]
    scores = np.zeros(len(db_clip_stack), dtype=np.float32)
    for i, cf in enumerate(db_clip_stack):
        if cf is not None and len(cf) > 0:
            scores[i] = float((cf @ q).max())
    return scores


def _ot_score(src_embs: np.ndarray, dst_embs: np.ndarray, reg: float = 0.05) -> float:
    """
    Sinkhorn OT cost between two sets of L2-normalised embeddings.
    Returns negative cost (higher = better match).
    Returns 0.0 if either set is empty or pot unavailable.
    """
    if pot is None or src_embs is None or dst_embs is None:
        return 0.0
    if len(src_embs) == 0 or len(dst_embs) == 0:
        return 0.0

    n, m = len(src_embs), len(dst_embs)
    p = np.ones(n, dtype=np.float64) / n
    q = np.ones(m, dtype=np.float64) / m

    # Cost matrix: semantic dissimilarity
    sim = _l2norm(src_embs) @ _l2norm(dst_embs).T   # (n, m)
    M   = np.clip(1.0 - sim, 0.0, 2.0).astype(np.float64)

    try:
        cost = float(pot.sinkhorn2(p, q, M, reg=reg))
    except Exception:
        cost = 1.0

    return -cost   # negate: higher score = lower OT cost = better match


def _gw_score(
    query_dets: list[dict],
    db_node_entry: dict,
    reg: float = 0.05,
) -> float:
    """
    Gromov-Wasserstein score between query detection graph and DB subgraph.

    Query graph: detections at time T, distances = L2 in normalised bbox-center space.
    DB graph   : scene-graph nodes, distances = 2D Euclidean in XY world space.
    Returns negative GW cost (higher = more structurally similar).
    """
    if pot is None:
        return 0.0

    node_pos = db_node_entry.get("positions")
    if node_pos is None or len(node_pos) == 0 or len(query_dets) == 0:
        return 0.0

    # ── Query intra-graph distance matrix (bbox centres in [0,1]^2) ──
    centers_q = np.array([
        [(d["bbox_norm"][0] + d["bbox_norm"][2]) / 2,
         (d["bbox_norm"][1] + d["bbox_norm"][3]) / 2]
        for d in query_dets
    ], dtype=np.float32)
    n = len(centers_q)
    D_q = np.linalg.norm(centers_q[:, None] - centers_q[None, :], axis=-1).astype(np.float64)
    D_q /= D_q.max() + 1e-12

    # ── DB intra-graph distance matrix (2D XY world) ──
    pos2d = node_pos[:, :2].astype(np.float64)
    m = len(pos2d)
    D_db = np.linalg.norm(pos2d[:, None] - pos2d[None, :], axis=-1)
    D_db /= D_db.max() + 1e-12

    p = np.ones(n, dtype=np.float64) / n
    q = np.ones(m, dtype=np.float64) / m

    try:
        gw_cost = float(pot.gromov_wasserstein2(D_q, D_db, p, q,
                                                 loss_fun="square_loss",
                                                 log=False))
    except Exception:
        gw_cost = 1.0

    return -gw_cost


def _snap_loc_rescore(
    vlad_scores: np.ndarray,
    t: int,
    ts: np.ndarray,
    siglip_encoded: dict,
    db_clip_avg: np.ndarray,
    alpha: float,
    top_k_candidates: int = 100,
) -> np.ndarray:
    """
    SnapLoc: combine DinoVLAD image scores with SigLIP im2graph scores.
    Only re-scores the top_k_candidates from image retrieval for speed.
    """
    scores = vlad_scores.copy()

    # Query SigLIP embedding at t
    X_siglip = siglip_encoded["clip_cls"]["X"][siglip_encoded["clip_cls"]["ts"] == t]
    if len(X_siglip) == 0:
        return scores

    q_clip = _l2norm(X_siglip.mean(axis=0)[None])[0]   # aggregate views

    # Graph re-score (avg mode) only for top candidates
    top_candidates = np.argsort(-vlad_scores)[:top_k_candidates]
    graph_s = np.zeros(len(vlad_scores), dtype=np.float32)
    graph_s[top_candidates] = (db_clip_avg[top_candidates] @ q_clip)

    # Normalise each component to [0,1] before blending
    def _minmax(x):
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo + 1e-12)

    scores = (1 - alpha) * _minmax(vlad_scores) + alpha * _minmax(graph_s)
    return scores


# ═══════════════════════════════════════════════════════════════════════════════
# Main experiment runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment(args):
    # ── 1. Load data ──────────────────────────────────────────────────────────
    print(f"Loading dataset from {args.dataset} …")
    with open(args.dataset, "rb") as f:
        dataset = pkl.load(f)

    db_traj          = np.array(dataset["db_traj"])       # (N_db, 3)
    query_traj       = np.array(dataset["query_traj"])    # (T, 3)
    db_room_labels   = dataset.get("db_room_labels", ["unknown"] * len(db_traj))
    query_room_labels = dataset.get("query_room_labels", ["unknown"] * len(query_traj))
    ts               = np.array(dataset["ts"])
    T                = int(ts[-1]) + 1

    print(f"  DB frames  : {len(db_traj)}")
    print(f"  Query steps: {T}")

    graph = {}
    if args.graph:
        print(f"Loading scene graph from {args.graph} …")
        graph = load_scene_graph(args.graph)
        print(f"  Nodes: {len(graph.get('nodes', []))}  Edges: {len(graph.get('edges', []))}")

    # ── 2. Decide which methods to run ────────────────────────────────────────
    requested = set(ALL_METHODS if "all" in args.methods else args.methods)
    needs_dino   = bool(requested & IM2IM_DINO_METHODS)
    needs_siglip = bool(requested & IM2IM_CLIP_METHODS)
    needs_graph  = bool(requested & ({"clip_avg", "clip_max", "dino_graph",
                                       "snap_loc"} | GRAPH2GRAPH_METHODS))
    needs_yolo   = bool(requested & GRAPH2GRAPH_METHODS)

    # ── 3. Load models ────────────────────────────────────────────────────────
    dino   = DinoModel(cfg)   if needs_dino   else None
    siglip = SiglipModel(cfg) if needs_siglip else None
    yolo   = YOLODetector(cfg) if needs_yolo  else None

    # ── 4. Pre-compute im2im embeddings ───────────────────────────────────────
    db_images    = dataset["db_images"]
    query_images = dataset["query_images"]

    dino_methods = {}
    if needs_dino:
        if "dino_cls" in requested or "dino_graph" in requested:
            dino_methods["dino_cls"]  = {"patches": False, "agg_method": "gap", "num_clusters": 32, "grayscale": False}
        if "dino_vlad" in requested or "snap_loc" in requested:
            dino_methods["dino_vlad"] = {"patches": True,  "agg_method": "vlad", "num_clusters": 32, "grayscale": False}

    clip_methods = {}
    if needs_siglip:
        if "clip_cls" in requested or "clip_avg" in requested or "clip_max" in requested \
                or "bag_of_objects" in requested or "bag_of_texts" in requested or "ot_gw" in requested:
            clip_methods["clip_cls"]  = {"patches": False, "agg_method": "gap", "num_clusters": 32, "grayscale": False}
        if "clip_vlad" in requested:
            clip_methods["clip_vlad"] = {"patches": True,  "agg_method": "vlad", "num_clusters": 32, "grayscale": False}

    dino_encoded = {}
    if dino_methods:
        print("Pre-computing DINO embeddings …")
        dino_encoded = prepare_embeddings(
            dino, db_images, query_images,
            methods=dino_methods, ts=ts, cropping=False, grayscale=False,
        )

    siglip_encoded = {}
    if clip_methods:
        print("Pre-computing SigLIP embeddings …")
        siglip_encoded = prepare_embeddings(
            siglip, db_images, query_images,
            methods=clip_methods, ts=ts, cropping=False, grayscale=False,
        )

    # ── 5. Build DB → graph node index ────────────────────────────────────────
    db_node_index = []
    db_clip_avg   = None
    db_clip_stack = None
    db_text_stack = None

    if needs_graph and graph.get("nodes"):
        print("Building DB frame → scene-graph node index …")
        db_node_index = build_db_node_index(db_traj, graph, window_m=args.graph_window)
        db_clip_avg, db_clip_stack = precompute_db_clip_aggregates(db_node_index)
        db_text_stack = precompute_db_text_stack(db_node_index)
    elif needs_graph:
        warnings.warn("Scene graph is empty — im2graph / graph2graph will return zero scores.")
        db_node_index = [{"clip_ft": None, "text_ft": None, "positions": np.zeros((0,3))}
                         for _ in range(len(db_traj))]
        db_clip_avg   = np.zeros((len(db_traj), 1), dtype=np.float32)
        db_clip_stack = [None] * len(db_traj)
        db_text_stack = [None] * len(db_traj)

    # ── 6. Build YOLO cache ────────────────────────────────────────────────────
    yolo_cache = None
    if needs_yolo:
        yolo_cache = load_or_build_yolo_cache(
            dataset, yolo, siglip,
            cache_path=args.yolo_cache or "",
        )

    # ── 7. Register score functions ────────────────────────────────────────────
    # Each function takes (t: int) and returns (N_db,) float32 scores.

    ground_truth = dataset["ground_truth"]

    def _im2im_score(encoded, method_key, t):
        scores, _, _ = localize_at_t(encoded, ground_truth, method_key, t, n_views=10)
        return np.array(scores, dtype=np.float32)

    def _im2graph_score(t, mode: str, encoded_clip, encoded_dino) -> np.ndarray:
        # Aggregate query embeddings at time t across all camera views
        if mode in ("clip_avg", "clip_max"):
            X = encoded_clip["clip_cls"]["X"][encoded_clip["clip_cls"]["ts"] == t]
        else:  # dino_graph
            X = encoded_dino["dino_cls"]["X"][encoded_dino["dino_cls"]["ts"] == t]

        if len(X) == 0:
            return np.zeros(len(db_traj), dtype=np.float32)

        q = _l2norm(X.mean(axis=0)[None])[0]

        if mode == "clip_avg":
            return _score_im2graph_avg(q, db_clip_avg)
        elif mode == "clip_max":
            return _score_im2graph_max(q, db_clip_stack)
        else:  # dino_graph — cross modal
            return _score_im2graph_avg(q, db_clip_avg)

    def _graph2graph_score(t, mode: str) -> np.ndarray:
        if yolo_cache is None:
            return np.zeros(len(db_traj), dtype=np.float32)

        dets = yolo_cache.get(t, [])
        if len(dets) == 0:
            return np.zeros(len(db_traj), dtype=np.float32)

        scores = np.zeros(len(db_traj), dtype=np.float32)

        if mode == "bag_of_objects":
            src_embs = np.stack([d["clip_ft"] for d in dets], axis=0)
            for i, dst_cf in enumerate(db_clip_stack):
                scores[i] = _ot_score(src_embs, dst_cf)

        elif mode == "bag_of_texts":
            class_names = list({d["class_name"] for d in dets})
            if not class_names:
                return scores
            text_embs = siglip.embed_texts(class_names)   # (K, D_text)
            if text_embs is None:
                return scores
            text_embs = _l2norm(text_embs)
            for i, dst_tf in enumerate(db_text_stack):
                scores[i] = _ot_score(text_embs, dst_tf)

        elif mode == "ot_gw":
            # Only score top-100 candidates from DinoVLAD for speed
            if "dino_vlad" in dino_encoded:
                pre_scores = np.array(_im2im_score(dino_encoded, "dino_vlad", t))
                candidates = np.argsort(-pre_scores)[:100]
            else:
                candidates = np.arange(len(db_traj))

            for i in candidates:
                scores[i] = _gw_score(dets, db_node_index[i])

        return scores

    method_fns = {}

    if "dino_cls"  in requested: method_fns["dino_cls"]  = lambda t: _im2im_score(dino_encoded,   "dino_cls",  t)
    if "clip_cls"  in requested: method_fns["clip_cls"]  = lambda t: _im2im_score(siglip_encoded, "clip_cls",  t)
    if "dino_vlad" in requested: method_fns["dino_vlad"] = lambda t: _im2im_score(dino_encoded,   "dino_vlad", t)
    if "clip_vlad" in requested: method_fns["clip_vlad"] = lambda t: _im2im_score(siglip_encoded, "clip_vlad", t)

    if "snap_loc"  in requested:
        method_fns["snap_loc"] = lambda t: _snap_loc_rescore(
            _im2im_score(dino_encoded, "dino_vlad", t),
            t, ts, siglip_encoded, db_clip_avg,
            alpha=args.snap_alpha,
        )

    if "clip_avg"  in requested: method_fns["clip_avg"]  = lambda t: _im2graph_score(t, "clip_avg",  siglip_encoded, dino_encoded)
    if "clip_max"  in requested: method_fns["clip_max"]  = lambda t: _im2graph_score(t, "clip_max",  siglip_encoded, dino_encoded)
    if "dino_graph" in requested: method_fns["dino_graph"] = lambda t: _im2graph_score(t, "dino_graph", siglip_encoded, dino_encoded)

    if "bag_of_objects" in requested: method_fns["bag_of_objects"] = lambda t: _graph2graph_score(t, "bag_of_objects")
    if "bag_of_texts"   in requested: method_fns["bag_of_texts"]   = lambda t: _graph2graph_score(t, "bag_of_texts")
    if "ot_gw"          in requested: method_fns["ot_gw"]          = lambda t: _graph2graph_score(t, "ot_gw")

    # Warn about methods that couldn't be set up
    for m in requested - set(method_fns.keys()):
        print(f"[WARN] Method '{m}' could not be set up and will be skipped.")

    # ── 8. Evaluation loop ────────────────────────────────────────────────────
    print(f"\nRunning evaluation over {T} timesteps × {len(method_fns)} methods …\n")

    per_t: dict[str, list[dict]] = {m: [] for m in method_fns}
    pred_positions: dict[str, list] = {m: [] for m in method_fns}
    gt_pos_list: list = []

    for t in tqdm(range(T), desc="Evaluation"):
        gt_pos  = query_traj[t]
        gt_room = query_room_labels[t] if t < len(query_room_labels) else "unknown"
        gt_pos_list.append(gt_pos)

        for method, score_fn in method_fns.items():
            scores = score_fn(t)

            pred_pos  = predict_position(scores, db_traj, k=args.top_k)
            pred_room = db_room_labels[int(np.argmax(scores))] if len(scores) > 0 else "unknown"

            per_t[method].append(compute_loc_metrics(pred_pos, gt_pos, pred_room, gt_room))
            pred_positions[method].append(pred_pos)

    # ── 9. Aggregate & print ──────────────────────────────────────────────────
    aggregated = {m: aggregate_results(per_t[m]) for m in method_fns}
    print(format_results_table(aggregated))

    # ── 10. Save outputs ──────────────────────────────────────────────────────
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    save_results_csv(aggregated, str(out / "results.csv"))

    with open(out / "per_timestep.pkl", "wb") as f:
        pkl.dump({"per_t": per_t, "pred_positions": pred_positions,
                  "gt_positions": gt_pos_list}, f)
    print(f"[saved] per-timestep data → {out / 'per_timestep.pkl'}")

    # ── 11. Heatmaps (optional) ───────────────────────────────────────────────
    gt_arr = np.array(gt_pos_list)
    for method in method_fns:
        pred_arr = np.array(pred_positions[method])
        errs     = np.array([r["err_m"] for r in per_t[method]])
        plot_heatmap(
            method_name=method,
            pred_positions=pred_arr,
            gt_positions=gt_arr,
            errors_m=errs,
            floorplan_path=args.floorplan or None,
            mapping_txt=args.mapping or None,
            output_path=str(out / f"heatmap_{method}.png"),
        )

    print(f"\nAll results saved to {out}/")
    return aggregated


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(
        description="Localization method comparison experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset",      required=True,
                   help="Path to spot_dataset_w_gt_*.pkl (from build_dataset.py)")
    p.add_argument("--graph",        default=None,
                   help="Path to scene_graph.json (required for im2graph / graph2graph)")
    p.add_argument("--mapping",      default="./mapping.txt",
                   help="Path to mapping.txt — used for floorplan coordinate mapping")
    p.add_argument("--floorplan",    default=None,
                   help="Path to floorplan.png (optional, for heatmap overlay)")
    p.add_argument("--yolo_cache",   default=None,
                   help="Path to YOLO detection cache pkl (built on first run if absent)")
    p.add_argument("--methods",      nargs="+", default=["all"],
                   choices=ALL_METHODS + ["all"],
                   help="Which methods to run. Pass 'all' for everything.")
    p.add_argument("--top_k",        type=int, default=10,
                   help="Top-k DB positions for weighted-average position prediction")
    p.add_argument("--graph_window", type=float, default=3.0,
                   help="Spatial window (m) for associating DB frames with graph nodes")
    p.add_argument("--snap_alpha",   type=float, default=0.4,
                   help="SnapLoc blending weight: alpha × graph_score + (1-alpha) × image_score")
    p.add_argument("--output_dir",   default="results",
                   help="Directory to save results.csv, heatmaps, per_timestep.pkl")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_experiment(args)
