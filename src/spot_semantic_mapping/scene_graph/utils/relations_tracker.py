from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Callable, Optional

import numpy as np
from scipy.spatial import cKDTree
import cv2


# ============================================================
# Data class
# ============================================================

@dataclass
class RelationEdge:
    src_id: int
    dst_id: int
    src: str
    dst: str
    rtype: str
    score: float
    dist: float


# ============================================================
# Allowed relation set (constrained) + inverses
# ============================================================

ALLOWED_RELATIONS = [
    "left_of",
    "right_of",
    "in_front_of",
    "behind",
    "above",
    "below",
    "inside",
    "contains",
    "on_top_of",
    "under",
    "next_to",
    "near",
    "overlapping",
    "no_relation",
]

INVERSE_REL = {
    "left_of": "right_of",
    "right_of": "left_of",
    "in_front_of": "behind",
    "behind": "in_front_of",
    "above": "below",
    "below": "above",
    "inside": "contains",
    "contains": "inside",
    "on_top_of": "under",
    "under": "on_top_of",
    "next_to": "next_to",
    "near": "near",
    "overlapping": "overlapping",
    "no_relation": "no_relation",
}

SYNONYM_MAP = {
    "to the left of": "left_of",
    "left of": "left_of",
    "left": "left_of",
    "to the right of": "right_of",
    "right of": "right_of",
    "right": "right_of",
    "in front of": "in_front_of",
    "front of": "in_front_of",
    "behind": "behind",
    "back of": "behind",
    "above": "above",
    "below": "below",
    "under": "under",
    "underneath": "under",
    "on top of": "on_top_of",
    "on_top_of": "on_top_of",
    "inside": "inside",
    "in": "inside",
    "contains": "contains",
    "next to": "next_to",
    "next_to": "next_to",
    "near": "near",
    "close to": "near",
    "overlapping": "overlapping",
    "overlap": "overlapping",
    "no relation": "no_relation",
    "no meaningful relation": "no_relation",
    "no_relation": "no_relation",
}


def normalize_relation(pred: str | None) -> str:
    if pred is None:
        return "no_relation"
    p = pred.strip().lower()
    p = " ".join(p.split())
    if p in INVERSE_REL:
        return p
    if p in SYNONYM_MAP:
        return SYNONYM_MAP[p]

    # very light keyword fallback
    if "left" in p: return "left_of"
    if "right" in p: return "right_of"
    if "front" in p: return "in_front_of"
    if "behind" in p or "back" in p: return "behind"
    if "above" in p or "over" in p: return "above"
    if "below" in p: return "below"
    if "under" in p: return "under"
    if "top" in p: return "on_top_of"
    if "inside" in p: return "inside"
    if "contain" in p: return "contains"
    if "next" in p: return "next_to"
    if "near" in p or "close" in p: return "near"
    if "overlap" in p or "intersect" in p: return "overlapping"
    return "no_relation"


# ============================================================
# Crop utils
# ============================================================

def draw_bounding_box_and_crop_with_labels(
    full_frame: np.ndarray,
    xyxyA, xyxyB,
    labelA="A",
    labelB="B",
    pad=10,
    colorA=(255, 0, 0),
    colorB=(0, 255, 0),
    thickness=3,
    font_scale=0.7,
    font_thickness=2,
) -> np.ndarray:
    x1 = min(xyxyA[0], xyxyB[0]) - pad
    y1 = min(xyxyA[1], xyxyB[1]) - pad
    x2 = max(xyxyA[2], xyxyB[2]) + pad
    y2 = max(xyxyA[3], xyxyB[3]) + pad

    H, W = full_frame.shape[:2]
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(W, int(x2))
    y2 = min(H, int(y2))

    crop = full_frame[y1:y2, x1:x2].copy()

    Ax1, Ay1, Ax2, Ay2 = map(int, xyxyA)
    Bx1, By1, Bx2, By2 = map(int, xyxyB)

    Ax1 -= x1; Ax2 -= x1
    Ay1 -= y1; Ay2 -= y1
    Bx1 -= x1; Bx2 -= x1
    By1 -= y1; By2 -= y1

    cv2.rectangle(crop, (Ax1, Ay1), (Ax2, Ay2), colorA, thickness)
    cv2.rectangle(crop, (Bx1, By1), (Bx2, By2), colorB, thickness)

    font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_label(img, text, x, y, color):
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        ty = max(y - 5, th + 5)
        cv2.rectangle(img, (x, ty - th - 4), (x + tw + 4, ty + 4), (0, 0, 0), -1)
        cv2.putText(img, text, (x + 2, ty - 2), font, font_scale, color, font_thickness, lineType=cv2.LINE_AA)

    draw_label(crop, str(labelA), Ax1, Ay1, colorA)
    draw_label(crop, str(labelB), Bx1, By1, colorB)
    return crop


# ============================================================
# (2) KNN candidate pairs (~3 neighbors/node)
# ============================================================

def build_knn_candidate_pairs(objects, k_per_node=3, max_radius_m=2.0):
    """
    objects[i] must have:
      - obj.bbox.get_center() -> (3,)
    Returns:
      pairs: set of (i,j) where i<j are *indices into objects*
      dist_ij: dict[(i,j)] -> centroid distance
    """
    cents = []
    valid = []
    for i, o in enumerate(objects):
        bbox = getattr(o, "bbox", None)
        if bbox is None:
            continue
        c = np.asarray(bbox.get_center(), dtype=np.float32).reshape(3,)
        if not np.all(np.isfinite(c)):
            continue
        cents.append(c)
        valid.append(i)

    if len(valid) <= 1:
        return set(), {}

    C = np.stack(cents, axis=0)
    tree = cKDTree(C)

    kk = min(k_per_node + 1, C.shape[0])  # +1 for self
    dists, nbrs = tree.query(C, k=kk)

    pairs: Set[Tuple[int, int]] = set()
    dist_ij: Dict[Tuple[int, int], float] = {}

    for a_local in range(C.shape[0]):
        a_global = valid[a_local]
        for t in range(1, kk):  # skip self
            b_local = int(nbrs[a_local][t])
            d = float(dists[a_local][t])
            if (not np.isfinite(d)) or (d > max_radius_m):
                continue
            b_global = valid[b_local]
            i, j = (a_global, b_global) if a_global < b_global else (b_global, a_global)
            pairs.add((i, j))
            if (i, j) not in dist_ij or d < dist_ij[(i, j)]:
                dist_ij[(i, j)] = d

    return pairs, dist_ij


def near_score(dist_m, sigma=0.6):
    return float(np.exp(-dist_m / (sigma + 1e-8)))


def make_base_near_edges(objects, pairs, dist_ij):
    """
    Creates bidirectional 'near' edges for candidate pairs.
    Uses objects[i].oid and objects[i].class_name.
    """
    edges: List[RelationEdge] = []
    for (i, j) in pairs:
        d = float(dist_ij.get((i, j), 0.0))
        s = near_score(d, sigma=0.6)
        oi, oj = objects[i], objects[j]
        edges.append(RelationEdge(oi.oid, oj.oid, oi.class_name, oj.class_name, "near", s, d))
        edges.append(RelationEdge(oj.oid, oi.oid, oj.class_name, oi.class_name, "near", s, d))
    return edges


# ============================================================
# Co-visibility index
# ============================================================

def build_obj_frames(objects):
    """
    obj_frames[obj_idx][frame_idx] = list[(xyxy, score)]
    Expects obj.crops entries:
      (crop, xyxy, fidx) or (crop, xyxy, fidx, score)
    """
    obj_frames = defaultdict(lambda: defaultdict(list))
    for obj_idx, obj in enumerate(objects):
        for item in getattr(obj, "crops", []):
            if item is None:
                continue
            if len(item) == 4:
                _, xyxy, fidx, score = item
            elif len(item) == 3:
                _, xyxy, fidx = item
                score = 1.0
            else:
                continue
            xyxy = np.array(xyxy).astype(int).tolist()
            obj_frames[obj_idx][int(fidx)].append((xyxy, float(score)))
    return obj_frames


def best_bbox_in_frame(obj_frames, obj_idx, fidx):
    lst = obj_frames[obj_idx].get(fidx, None)
    if not lst:
        return None
    lst = sorted(lst, key=lambda x: x[1], reverse=True)
    return lst[0][0]


def best_common_frames(obj_frames, objA, objB, top_k=3):
    common = set(obj_frames[objA].keys()) & set(obj_frames[objB].keys())
    if not common:
        return []
    scored = []
    for f in common:
        scoreA = max(s for (_, s) in obj_frames[objA][f])
        scoreB = max(s for (_, s) in obj_frames[objB][f])
        scored.append((f, scoreA + scoreB))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [f for (f, _) in scored[:top_k]]


def majority_vote_relation(votes):
    votes = [normalize_relation(v) for v in votes]
    votes = [v for v in votes if v in INVERSE_REL]
    if not votes:
        return "no_relation"
    return Counter(votes).most_common(1)[0][0]


# ============================================================
# VLM refinement
# ============================================================

def refine_edges_with_vlm(
    objects,
    candidate_pairs,
    dist_ij,
    frame_loader: Callable[[int], np.ndarray],
    vlm_model: Callable[..., str],
    top_k_frames=3,
    pad=12,
):
    """
    Returns:
      refined_map: dict[(src_oid, dst_oid)] = RelationEdge (directed)
    """
    obj_frames = build_obj_frames(objects)

    allowed_str = ", ".join([r for r in ALLOWED_RELATIONS])  # include no_relation

    refined_map: Dict[Tuple[int, int], RelationEdge] = {}

    for (i, j) in candidate_pairs:
        frames = best_common_frames(obj_frames, i, j, top_k=top_k_frames)
        if not frames:
            continue

        oi, oj = objects[i], objects[j]
        Aname, Bname = oi.class_name, oj.class_name
        votes = []

        for fidx in frames:
            frame = frame_loader(fidx)

            xyxyA = best_bbox_in_frame(obj_frames, i, fidx)
            xyxyB = best_bbox_in_frame(obj_frames, j, fidx)
            if xyxyA is None or xyxyB is None:
                continue

            pair_img = draw_bounding_box_and_crop_with_labels(
                frame, xyxyA, xyxyB, labelA=Aname, labelB=Bname, pad=pad
            )

            prompt = f"""
You are given an image with EXACTLY TWO labeled objects:
- Object A (red box): {Aname}
- Object B (green box): {Bname}

Choose the SINGLE best spatial relation describing A RELATIVE TO B.

You MUST output EXACTLY ONE label from this set:
{allowed_str}

Rules:
- left_of/right_of/in_front_of/behind are w.r.t. the IMAGE viewpoint.
- above/below/on_top_of/under for vertical relations.
- inside/contains only if clearly true.
- next_to/near/overlapping if appropriate.
- If unclear, output: no_relation

Return ONLY the label.
""".strip()

            raw = vlm_model(prompt, image=pair_img)
            votes.append(raw)

        rel = majority_vote_relation(votes)
        if rel == "no_relation":
            continue

        d = float(dist_ij.get((min(i, j), max(i, j)), 0.0))
        s = 1.0  # placeholder; you can replace with self-consistency score

        refined_map[(oi.oid, oj.oid)] = RelationEdge(oi.oid, oj.oid, Aname, Bname, rel, s, d)

        inv = INVERSE_REL.get(rel, "no_relation")
        if inv != "no_relation":
            refined_map[(oj.oid, oi.oid)] = RelationEdge(oj.oid, oi.oid, Bname, Aname, inv, s, d)

    return refined_map


# ============================================================
# (4) Merge base edges + refined edges (overwrite near)
# ============================================================

def merge_edges(base_edges: List[RelationEdge],
                refined_map: Dict[Tuple[int, int], RelationEdge],
                overwrite_near: bool = True) -> List[RelationEdge]:
    out: List[RelationEdge] = []
    for e in base_edges:
        key = (e.src_id, e.dst_id)
        if key in refined_map:
            if overwrite_near:
                out.append(refined_map[key])
            else:
                out.append(e)
                out.append(refined_map[key])
        else:
            out.append(e)
    return out


def report_degree_stats(edges: List[RelationEdge], objects, logger=None):
    deg_out = defaultdict(int)
    for e in edges:
        deg_out[e.src_id] += 1
    degs = [deg_out[getattr(o, "oid")] for o in objects]
    msg = f"[RoomGraph] directed out-degree: min={min(degs)}, mean={np.mean(degs):.2f}, max={max(degs)}"
    if logger:
        logger.info(msg)
    else:
        print(msg)


# ============================================================
# FULL PIPELINE
# ============================================================

def build_sparse_scene_graph_edges(
    tracker,
    frame_loader: Callable[[int], np.ndarray],
    vlm_model: Callable[..., str],
    logger=None,
    neighbors_per_node=3,
    max_radius_m=2.0,
    top_k_frames=3,
    pad=12,
):
    """
    Implements:
      - Solution 2: KNN candidate pairs (~3 neighbors per node)
      - Solution 4: Merge refined edges onto base near scaffold (overwrite near)

    Returns: list[RelationEdge] (directed), ~2*k outgoing total if bidirectional scaffold.
    """
    objects = tracker.objects
    if logger:
        logger.info(f"[Graph] Building candidates for {len(objects)} objects...")

    # (2) KNN candidate undirected pairs
    pairs, dist_ij = build_knn_candidate_pairs(
        objects,
        k_per_node=neighbors_per_node,
        max_radius_m=max_radius_m
    )
    if logger:
        target = len(objects) * neighbors_per_node / 2.0
        logger.info(f"[Graph] Candidate undirected pairs: {len(pairs)} (target ~{target:.0f})")

    # Base scaffold: bidirectional near edges
    base_edges = make_base_near_edges(objects, pairs, dist_ij)

    # VLM refine where co-visible
    refined_map = refine_edges_with_vlm(
        objects=objects,
        candidate_pairs=pairs,
        dist_ij=dist_ij,
        frame_loader=frame_loader,
        vlm_model=vlm_model,
        top_k_frames=top_k_frames,
        pad=pad,
    )
    if logger:
        logger.info(f"[Graph] Refined directed edges: {len(refined_map)}")

    # (4) Merge: overwrite 'near' with refined labels
    edges = merge_edges(base_edges, refined_map, overwrite_near=True)

    # Store
    tracker.edges = edges

    report_degree_stats(edges, objects, logger=logger)
    return edges
