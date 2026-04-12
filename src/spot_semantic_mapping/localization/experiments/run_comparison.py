"""
run_comparison.py
=================
Paper experiment: compare 10 localization methods across three families on
the SPOT indoor dataset.

Families & methods
------------------
im2im (→ position via image retrieval over DB frames)
  - dino_cls        : DINOv2 CLS cosine retrieval
  - clip_cls        : SigLIP CLS cosine retrieval
  - dino_vlad       : DinoVLAD (AnyLoc) — patch VLAD
  - clip_vlad       : ClipVLAD — SigLIP patch VLAD
  - snap_loc        : DinoVLAD + SigLIP graph re-scoring (SnapLoc)

im2graph (→ position via query image vs ALL scene-graph node embeddings)
  - clip_avg        : avg-views SigLIP CLS cosine sim vs all node clip_ft
  - clip_max        : max-over-views SigLIP CLS cosine sim vs all node clip_ft

graph2graph (→ position via YOLO detections vs ALL scene-graph node embeddings, OT)
  - bag_of_objects  : YOLO crop SigLIP embs vs all node clip_ft (Sinkhorn OT)
  - bag_of_texts    : YOLO class-name text embs vs all node text_ft (Sinkhorn OT)
  - ot_gw           : Gromov-Wasserstein structural matching

Metrics
-------
- R@1 / R@3 / R@5 / R@10 : retrieval recall (im2im only)
- 3m / 5m accuracy        : % timesteps with predicted pos within 3 / 5 m of GT
- room accuracy            : top-1 retrieved frame/node's room matches GT room
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
    "clip_avg", "clip_max",
    "bag_of_texts", "bag_of_objects", "ot_gw",
]

# im2graph + graph2graph: score fn returns (pred_pos, pred_room) directly from nodes
IM2GRAPH_METHODS = {"clip_avg", "clip_max", "bag_of_texts", "bag_of_objects", "ot_gw"}

GRAPH2GRAPH_METHODS = {"bag_of_texts", "bag_of_objects", "ot_gw"}

IM2IM_DINO_METHODS  = {"dino_cls", "dino_vlad", "snap_loc"}
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
    Used only for SnapLoc graph re-scoring (snap_loc method).

    Returns a list of length N_db. Each entry is a dict:
        {"clip_ft": (K, D) | None, "positions": (K, 3)}
    """
    nodes = graph.get("nodes", [])
    if len(nodes) == 0:
        return [{"clip_ft": None, "positions": np.zeros((0, 3))}
                for _ in range(len(db_traj))]

    node_pos = np.array([n["position"] for n in nodes], dtype=np.float32)
    clip_fts  = [n.get("clip_ft") for n in nodes]

    result = []
    for pos in db_traj:
        dists = np.linalg.norm(node_pos[:, :2] - pos[:2], axis=1)
        mask  = dists <= window_m
        nearby_pos  = node_pos[mask]
        valid_clip = [np.asarray(clip_fts[j], dtype=np.float32)
                      for j in np.where(mask)[0] if clip_fts[j] is not None]
        nearby_clip = _l2norm(np.stack(valid_clip, axis=0)) if valid_clip else None
        result.append({"clip_ft": nearby_clip, "positions": nearby_pos})

    return result


def precompute_db_clip_avg(db_node_index: list[dict]) -> np.ndarray:
    """
    Returns db_clip_avg : (N_db, D) — mean clip_ft per DB frame (zeros where no nodes).
    Used only for SnapLoc graph re-scoring.
    """
    first_clip = next((e["clip_ft"] for e in db_node_index if e["clip_ft"] is not None), None)
    D = first_clip.shape[-1] if first_clip is not None else 1

    db_clip_avg = np.zeros((len(db_node_index), D), dtype=np.float32)
    for i, entry in enumerate(db_node_index):
        cf = entry["clip_ft"]
        if cf is not None and len(cf) > 0:
            db_clip_avg[i] = cf.mean(axis=0)

    return _l2norm(db_clip_avg)


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

                class_name = (yolo_model.class_names[cid]
                              if cid < len(yolo_model.class_names) else "object")
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
# OT helper (module-level, no closures)
# ═══════════════════════════════════════════════════════════════════════════════

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

    sim = _l2norm(src_embs) @ _l2norm(dst_embs).T
    M   = np.clip(1.0 - sim, 0.0, 2.0).astype(np.float64)

    try:
        cost = float(pot.sinkhorn2(p, q, M, reg=reg))
    except Exception:
        cost = 1.0

    return -cost


def _gw_score(query_dets: list[dict], db_node_entry: dict, reg: float = 0.05) -> float:
    """
    Gromov-Wasserstein score between query detection graph and a node subgraph.
    Returns negative GW cost (higher = more structurally similar).
    """
    if pot is None:
        return 0.0

    node_pos = db_node_entry.get("positions")
    if node_pos is None or len(node_pos) == 0 or len(query_dets) == 0:
        return 0.0

    centers_q = np.array([
        [(d["bbox_norm"][0] + d["bbox_norm"][2]) / 2,
         (d["bbox_norm"][1] + d["bbox_norm"][3]) / 2]
        for d in query_dets
    ], dtype=np.float32)
    n = len(centers_q)
    D_q = np.linalg.norm(centers_q[:, None] - centers_q[None, :], axis=-1).astype(np.float64)
    D_q /= D_q.max() + 1e-12

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


# ═══════════════════════════════════════════════════════════════════════════════
# Main experiment runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment(args):
    # ── 1. Load data ──────────────────────────────────────────────────────────
    print(f"Loading dataset from {args.dataset} …")
    with open(args.dataset, "rb") as f:
        dataset = pkl.load(f)

    db_traj           = np.array(dataset["db_traj"])       # (N_db, 3)
    query_traj        = np.array(dataset["query_traj"])    # (T, 3)
    db_room_labels    = dataset.get("db_room_labels",    ["unknown"] * len(db_traj))
    query_room_labels = dataset.get("query_room_labels", ["unknown"] * len(query_traj))
    ts                = np.array(dataset["ts"])
    T                 = int(ts[-1]) + 1

    # Build fast GT lookup: timestep t → set of matching DB frame indices
    ground_truth = dataset.get("ground_truth")
    gt_sets: dict[int, set] = {}
    if ground_truth is not None:
        for row in ground_truth:
            gt_sets[int(row[0])] = set(int(i) for i in row[1])

    print(f"  DB frames  : {len(db_traj)}")
    print(f"  Query steps: {T}")

    # ── 2. Load scene graph ───────────────────────────────────────────────────
    graph: dict = {}
    if args.graph:
        print(f"Loading scene graph from {args.graph} …")
        graph = load_scene_graph(args.graph)
        print(f"  Nodes: {len(graph.get('nodes', []))}  Edges: {len(graph.get('edges', []))}")

    # Flatten ALL graph nodes for im2graph / graph2graph scoring
    all_clip_nodes = [n for n in graph.get("nodes", []) if n.get("clip_ft") is not None]
    all_text_nodes = [n for n in graph.get("nodes", []) if n.get("text_ft") is not None]

    all_node_clip_ft = (
        _l2norm(np.stack([n["clip_ft"] for n in all_clip_nodes], axis=0).astype(np.float32))
        if all_clip_nodes else np.zeros((0, 1), dtype=np.float32)
    )  # (N, D_clip)
    all_node_text_ft = (
        _l2norm(np.stack([n["text_ft"] for n in all_text_nodes], axis=0).astype(np.float32))
        if all_text_nodes else np.zeros((0, 1), dtype=np.float32)
    )  # (M, D_text)
    all_node_clip_pos = (
        np.stack([n["position"] for n in all_clip_nodes], axis=0).astype(np.float32)
        if all_clip_nodes else np.zeros((0, 3), dtype=np.float32)
    )  # (N, 3)
    all_node_text_pos = (
        np.stack([n["position"] for n in all_text_nodes], axis=0).astype(np.float32)
        if all_text_nodes else np.zeros((0, 3), dtype=np.float32)
    )  # (M, 3)
    all_node_clip_rooms = [n.get("room", "unknown") for n in all_clip_nodes]
    all_node_text_rooms = [n.get("room", "unknown") for n in all_text_nodes]

    if all_clip_nodes:
        print(f"  Node clip_ft arrays: {len(all_clip_nodes)} nodes, dim={all_node_clip_ft.shape[-1]}")
    if all_text_nodes:
        print(f"  Node text_ft arrays: {len(all_text_nodes)} nodes, dim={all_node_text_ft.shape[-1]}")

    # ── 3. Decide which methods to run ────────────────────────────────────────
    requested    = set(ALL_METHODS if "all" in args.methods else args.methods)
    needs_dino   = bool(requested & IM2IM_DINO_METHODS)
    needs_siglip = bool(requested & IM2IM_CLIP_METHODS)
    needs_graph  = bool(requested & ({"clip_avg", "clip_max", "snap_loc"} | GRAPH2GRAPH_METHODS))
    needs_yolo   = bool(requested & GRAPH2GRAPH_METHODS)

    # ── 4. Load models ────────────────────────────────────────────────────────
    dino   = DinoModel(cfg)    if needs_dino   else None
    siglip = SiglipModel(cfg)  if needs_siglip else None
    yolo   = YOLODetector(cfg) if needs_yolo   else None

    # ── 5. Pre-compute im2im embeddings ───────────────────────────────────────
    db_images    = dataset["db_images"]
    query_images = dataset["query_images"]

    dino_methods = {}
    if needs_dino:
        if "dino_cls" in requested:
            dino_methods["dino_cls"]  = {"patches": False, "agg_method": "gap",  "num_clusters": 32, "grayscale": False}
        if "dino_vlad" in requested or "snap_loc" in requested:
            dino_methods["dino_vlad"] = {"patches": True,  "agg_method": "vlad", "num_clusters": 32, "grayscale": False}

    clip_methods = {}
    if needs_siglip:
        if any(m in requested for m in ("clip_cls", "clip_avg", "clip_max",
                                         "bag_of_objects", "bag_of_texts",
                                         "ot_gw", "snap_loc")):
            clip_methods["clip_cls"]  = {"patches": False, "agg_method": "gap",  "num_clusters": 32, "grayscale": False}
        if "clip_vlad" in requested:
            clip_methods["clip_vlad"] = {"patches": True,  "agg_method": "vlad", "num_clusters": 32, "grayscale": False}

    dino_encoded: dict = {}
    if dino_methods:
        print("Pre-computing DINO embeddings …")
        dino_encoded = prepare_embeddings(
            dino, db_images, query_images,
            methods=dino_methods, ts=ts, cropping=False, grayscale=False,
        )

    siglip_encoded: dict = {}
    if clip_methods:
        print("Pre-computing SigLIP embeddings …")
        siglip_encoded = prepare_embeddings(
            siglip, db_images, query_images,
            methods=clip_methods, ts=ts, cropping=False, grayscale=False,
        )

    # ── 6. Build SnapLoc db_clip_avg (DB-frame level, for graph re-scoring) ──
    db_clip_avg: np.ndarray | None = None
    if "snap_loc" in requested and graph.get("nodes"):
        print("Building DB frame → scene-graph node index for SnapLoc …")
        db_node_index = build_db_node_index(db_traj, graph, window_m=args.graph_window)
        db_clip_avg   = precompute_db_clip_avg(db_node_index)
    elif "snap_loc" in requested:
        warnings.warn("Scene graph is empty — SnapLoc graph re-scoring will be zero.")
        db_clip_avg = np.zeros((len(db_traj), 1), dtype=np.float32)

    # ── 7. Build YOLO cache ────────────────────────────────────────────────────
    yolo_cache: dict | None = None
    if needs_yolo:
        yolo_cache = load_or_build_yolo_cache(
            dataset, yolo, siglip,
            cache_path=args.yolo_cache or "",
        )

    # ── 8. Define closures ─────────────────────────────────────────────────────

    def _predict_from_nodes(
        scores: np.ndarray,
        node_pos: np.ndarray,
        node_rooms: list,
        k: int = 10,
    ):
        """Softmax-weighted top-k node positions → (pred_pos (3,), pred_room str)."""
        k = min(k, len(scores))
        if k == 0:
            return np.zeros(3, dtype=np.float32), "unknown"
        top_k = np.argsort(-scores)[:k]
        raw_w = scores[top_k] - scores[top_k].max()
        w = np.exp(raw_w)
        w /= w.sum() + 1e-12
        pred_pos  = (w[:, None] * node_pos[top_k]).sum(axis=0)
        pred_room = node_rooms[top_k[0]]
        return pred_pos, pred_room

    def _im2im_score(encoded, method_key, t):
        scores, _, _ = localize_at_t(encoded, ground_truth, method_key, t, n_views=10)
        return np.array(scores, dtype=np.float32)

    def _snap_loc_rescore(vlad_scores: np.ndarray, t: int) -> np.ndarray:
        """SnapLoc: DinoVLAD scores blended with SigLIP im2graph scores over DB frames."""
        if db_clip_avg is None:
            return vlad_scores

        X = siglip_encoded["clip_cls"]["X"][siglip_encoded["clip_cls"]["ts"] == t]
        if len(X) == 0:
            return vlad_scores

        q_clip = _l2norm(X.mean(axis=0)[None])[0]

        top_candidates = np.argsort(-vlad_scores)[:100]
        graph_s = np.zeros(len(vlad_scores), dtype=np.float32)
        graph_s[top_candidates] = db_clip_avg[top_candidates] @ q_clip

        def _minmax(x):
            lo, hi = x.min(), x.max()
            return (x - lo) / (hi - lo + 1e-12)

        return (1 - args.snap_alpha) * _minmax(vlad_scores) + args.snap_alpha * _minmax(graph_s)

    def _im2graph_score(t: int, mode: str):
        """
        Compare query SigLIP CLS against ALL scene-graph node clip_ft.
        Returns (pred_pos (3,), pred_room str).
        """
        if len(all_node_clip_ft) == 0:
            return np.zeros(3, dtype=np.float32), "unknown"

        enc = siglip_encoded.get("clip_cls")
        if enc is None:
            return np.zeros(3, dtype=np.float32), "unknown"
        X = enc["X"][enc["ts"] == t]
        if len(X) == 0:
            return np.zeros(3, dtype=np.float32), "unknown"

        if mode == "clip_avg":
            q = _l2norm(X.mean(axis=0)[None])[0]
            scores = (all_node_clip_ft @ q).astype(np.float32)       # (N_nodes,)
        else:  # clip_max
            sims   = np.array(cosine_similarity_jax(X, all_node_clip_ft))  # (V, N_nodes)
            scores = sims.max(axis=0).astype(np.float32)               # (N_nodes,)

        return _predict_from_nodes(scores, all_node_clip_pos, all_node_clip_rooms, k=args.top_k)

    def _graph2graph_score(t: int, mode: str):
        """
        YOLO detections vs ALL scene-graph nodes via OT.
        Returns (pred_pos (3,), pred_room str).
        """
        if yolo_cache is None:
            return np.zeros(3, dtype=np.float32), "unknown"
        dets = yolo_cache.get(t, [])
        if len(dets) == 0:
            return np.zeros(3, dtype=np.float32), "unknown"

        if mode == "bag_of_objects":
            if len(all_node_clip_ft) == 0:
                return np.zeros(3, dtype=np.float32), "unknown"
            src_embs = _l2norm(np.stack([d["clip_ft"] for d in dets], axis=0))  # (K, D)
            # Full OT plan: query crops vs all nodes; per-node score = weighted transport mass
            C = np.clip(1.0 - (src_embs @ all_node_clip_ft.T), 0.0, 2.0).astype(np.float64)
            p = np.ones(len(src_embs), dtype=np.float64) / len(src_embs)
            q = np.ones(len(all_node_clip_ft), dtype=np.float64) / len(all_node_clip_ft)
            try:
                T_ot   = pot.sinkhorn(p, q, C, reg=0.05)
                # Per-node relevance: negative weighted OT cost (lower cost = higher relevance)
                scores = -np.array((T_ot * C).sum(axis=0), dtype=np.float32)
            except Exception:
                scores = (src_embs @ all_node_clip_ft.T).mean(axis=0).astype(np.float32)
            return _predict_from_nodes(scores, all_node_clip_pos, all_node_clip_rooms, k=args.top_k)

        elif mode == "bag_of_texts":
            if len(all_node_text_ft) == 0:
                return np.zeros(3, dtype=np.float32), "unknown"
            class_names = list({d["class_name"] for d in dets})
            text_embs = siglip.embed_texts(class_names)
            if text_embs is None or len(text_embs) == 0:
                return np.zeros(3, dtype=np.float32), "unknown"
            text_embs = _l2norm(np.array(text_embs, dtype=np.float32))
            C = np.clip(1.0 - (text_embs @ all_node_text_ft.T), 0.0, 2.0).astype(np.float64)
            p = np.ones(len(text_embs), dtype=np.float64) / len(text_embs)
            q = np.ones(len(all_node_text_ft), dtype=np.float64) / len(all_node_text_ft)
            try:
                T_ot   = pot.sinkhorn(p, q, C, reg=0.05)
                scores = -np.array((T_ot * C).sum(axis=0), dtype=np.float32)
            except Exception:
                scores = (text_embs @ all_node_text_ft.T).mean(axis=0).astype(np.float32)
            return _predict_from_nodes(scores, all_node_text_pos, all_node_text_rooms, k=args.top_k)

        elif mode == "ot_gw":
            if len(all_node_clip_ft) == 0:
                return np.zeros(3, dtype=np.float32), "unknown"
            # Pre-filter to top-100 nodes by DinoVLAD (for speed)
            if "dino_vlad" in dino_encoded:
                pre_scores = np.array(_im2im_score(dino_encoded, "dino_vlad", t))
                # Map DB-frame scores → node scores via spatial proximity
                node_pre = np.zeros(len(all_node_clip_ft), dtype=np.float32)
                for ni, npos in enumerate(all_node_clip_pos):
                    dists   = np.linalg.norm(db_traj[:, :2] - npos[:2], axis=1)
                    nearby  = np.where(dists <= args.graph_window)[0]
                    node_pre[ni] = pre_scores[nearby].max() if len(nearby) else 0.0
                candidates = np.argsort(-node_pre)[:100]
            else:
                candidates = np.arange(min(100, len(all_node_clip_ft)))

            gw_scores = np.full(len(all_node_clip_ft), -1.0, dtype=np.float32)
            for ni in candidates:
                gw_scores[ni] = _gw_score(dets, {"positions": all_node_clip_pos[ni:ni+1]})

            return _predict_from_nodes(gw_scores, all_node_clip_pos, all_node_clip_rooms, k=args.top_k)

        return np.zeros(3, dtype=np.float32), "unknown"

    # ── 9. Register method functions ───────────────────────────────────────────
    method_fns: dict = {}

    if "dino_cls"  in requested: method_fns["dino_cls"]  = lambda t: _im2im_score(dino_encoded,   "dino_cls",  t)
    if "clip_cls"  in requested: method_fns["clip_cls"]  = lambda t: _im2im_score(siglip_encoded, "clip_cls",  t)
    if "dino_vlad" in requested: method_fns["dino_vlad"] = lambda t: _im2im_score(dino_encoded,   "dino_vlad", t)
    if "clip_vlad" in requested: method_fns["clip_vlad"] = lambda t: _im2im_score(siglip_encoded, "clip_vlad", t)

    if "snap_loc"  in requested:
        method_fns["snap_loc"] = lambda t: _snap_loc_rescore(
            _im2im_score(dino_encoded, "dino_vlad", t), t
        )

    if "clip_avg"       in requested: method_fns["clip_avg"]       = lambda t: _im2graph_score(t, "clip_avg")
    if "clip_max"       in requested: method_fns["clip_max"]       = lambda t: _im2graph_score(t, "clip_max")
    if "bag_of_objects" in requested: method_fns["bag_of_objects"] = lambda t: _graph2graph_score(t, "bag_of_objects")
    if "bag_of_texts"   in requested: method_fns["bag_of_texts"]   = lambda t: _graph2graph_score(t, "bag_of_texts")
    if "ot_gw"          in requested: method_fns["ot_gw"]          = lambda t: _graph2graph_score(t, "ot_gw")

    for m in requested - set(method_fns.keys()):
        print(f"[WARN] Method '{m}' could not be set up and will be skipped.")

    # ── 10. Evaluation loop ───────────────────────────────────────────────────
    print(f"\nRunning evaluation over {T} timesteps × {len(method_fns)} methods …\n")

    per_t: dict[str, list[dict]] = {m: [] for m in method_fns}
    pred_positions: dict[str, list] = {m: [] for m in method_fns}
    gt_pos_list: list = []

    for t in tqdm(range(T), desc="Evaluation"):
        gt_pos  = query_traj[t]
        gt_room = query_room_labels[t] if t < len(query_room_labels) else "unknown"
        gt_set  = gt_sets.get(t, set())
        gt_pos_list.append(gt_pos)

        for method, score_fn in method_fns.items():
            if method in IM2GRAPH_METHODS:
                # Returns (pred_pos, pred_room) directly from node scoring
                pred_pos, pred_room = score_fn(t)
                scores = None   # R@K not applicable for im2graph/graph2graph
            else:
                # Returns (N_db,) scores → derive position + room from DB frames
                scores    = score_fn(t)
                pred_pos  = predict_position(scores, db_traj, k=args.top_k)
                pred_room = (db_room_labels[int(np.argmax(scores))]
                             if len(scores) > 0 else "unknown")

            metrics = compute_loc_metrics(
                pred_pos, gt_pos, pred_room, gt_room,
                scores=scores, gt_set=gt_set,
            )
            per_t[method].append(metrics)
            pred_positions[method].append(pred_pos)

    # ── 11. Aggregate & print ─────────────────────────────────────────────────
    aggregated = {m: aggregate_results(per_t[m]) for m in method_fns}
    print(format_results_table(aggregated))

    # ── 12. Save outputs ──────────────────────────────────────────────────────
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    save_results_csv(aggregated, str(out / "results.csv"))

    with open(out / "per_timestep.pkl", "wb") as f:
        pkl.dump({"per_t": per_t, "pred_positions": pred_positions,
                  "gt_positions": gt_pos_list}, f)
    print(f"[saved] per-timestep data → {out / 'per_timestep.pkl'}")

    # ── 13. Heatmaps (optional) ───────────────────────────────────────────────
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
                   help="Top-k positions for weighted-average position prediction")
    p.add_argument("--graph_window", type=float, default=3.0,
                   help="Spatial window (m) for SnapLoc DB-frame → node association")
    p.add_argument("--snap_alpha",   type=float, default=0.4,
                   help="SnapLoc blend weight: alpha × graph_score + (1-alpha) × image_score")
    p.add_argument("--output_dir",   default="results",
                   help="Directory to save results.csv, heatmaps, per_timestep.pkl")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_experiment(args)
