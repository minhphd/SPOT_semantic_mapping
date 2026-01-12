from collections import Counter, defaultdict
import numpy as np
from scipy.spatial import cKDTree
from dataclasses import dataclass
import json
import cv2

# -----------------------------------------
# Data class for relation edges
# -----------------------------------------
@dataclass
class RelationEdge:
    src_id: int
    dist_id: int
    src: str
    dst: str
    rtype: str
    score: float
    dist: float

def __repr__(self):
    return f"RelationEdge(src={self.src}, dst={self.dst}, rtype={self.rtype}, score={self.score:.3f}, dist={self.dist:.3f})"

# -----------------------------------------
# GEOMETRIC SDF METHOD
# -----------------------------------------
def unsigned_distance(A_pts: np.ndarray, B_pts: np.ndarray, k=1):
    """Minimum A→B distance using KDTree."""
    if len(A_pts) == 0 or len(B_pts) == 0:
        return np.inf
    tree = cKDTree(B_pts)
    d, _ = tree.query(A_pts, k=k)
    return float(np.min(d))


def inside_bbox(centroid, bbox_min, bbox_max):
    return np.all(centroid >= bbox_min) and np.all(centroid <= bbox_max)


def bbox_signed_dist_to(B_min, B_max, point):
    dx = max(B_min[0] - point[0], 0, point[0] - B_max[0])
    dy = max(B_min[1] - point[1], 0, point[1] - B_max[1])
    dz = max(B_min[2] - point[2], 0, point[2] - B_max[2])
    d = np.sqrt(dx*dx + dy*dy + dz*dz)
    return -d if inside_bbox(point, B_min, B_max) else d


def overlap_2D(A_min, A_max, B_min, B_max, axes):
    for ax in axes:
        if A_max[ax] < B_min[ax] or B_max[ax] < A_min[ax]:
            return False
    return True


def contact_score(dist, thresh=0.03, sigma=0.02):
    if dist > thresh:
        return 0.0
    return float(np.exp(-dist / sigma))


def near_score(cdist, max_dist=0.6, sigma=0.25):
    if cdist > max_dist:
        return 0.0
    return float(np.exp(-cdist / sigma))


def on_top_of_score(A, B, up_axis=2):
    """A on top of B if A bottom is slightly above B top + horizontal overlap."""
    A_min, A_max = A.bbox.get_min_bound(), A.bbox.get_max_bound()
    B_min, B_max = B.bbox.get_min_bound(), B.bbox.get_max_bound()

    # altitude ordering
    if A_min[up_axis] < B_max[up_axis] - 0.04:
        return 0.0

    # overlap in horizontal plane
    axes = [0,1,2]
    axes.remove(up_axis)
    if not overlap_2D(A_min, A_max, B_min, B_max, axes):
        return 0.0

    # must also be close
    d = unsigned_distance(np.asarray(A.pcd.points), np.asarray(B.pcd.points))
    return contact_score(d, thresh=0.04, sigma=0.02)


def inside_score(A, B):
    c = np.asarray(A.pcd.points).mean(axis=0)
    B_min, B_max = B.bbox.get_min_bound(), B.bbox.get_max_bound()
    d = bbox_signed_dist_to(B_min, B_max, c)
    if d >= 0.05:
        return 0.0
    return float(np.exp(-abs(d) / 0.02))


def compute_geometric_relations(objects, up_axis=2):
    """
    objects: tracker.objects (each obj has: oid, pcd, bbox)
    returns: list[RelationEdge]
    """
    edges = []

    N = len(objects)
    for i in range(N):
        Ai = objects[i]
        Ai_pts = np.asarray(Ai.pcd.points)
        Ai_cent = Ai_pts.mean(axis=0)

        for j in range(i+1, N):
            Bj = objects[j]
            Bj_pts = np.asarray(Bj.pcd.points)
            Bj_cent = Bj_pts.mean(axis=0)

            # -------------------------
            # DISTANCES
            # -------------------------
            dist = unsigned_distance(Ai_pts, Bj_pts)

            # centroid distance for NEAR
            cdist = np.linalg.norm(Ai_cent - Bj_cent)

            # -------------------------
            # CONTACT
            # -------------------------
            cscore = contact_score(dist)
            if cscore > 0:
                edges.append(RelationEdge(Ai.class_name, Bj.class_name, "CONTACT", cscore, dist))
                edges.append(RelationEdge(Bj.class_name, Ai.class_name, "CONTACT", cscore, dist))

            # -------------------------
            # NEAR
            # -------------------------
            nscore = near_score(cdist)
            if nscore > 0:
                edges.append(RelationEdge(Ai.class_name, Bj.class_name, "NEAR", nscore, cdist))
                edges.append(RelationEdge(Bj.class_name, Ai.class_name, "NEAR", nscore, cdist))

            # -------------------------
            # ON-TOP-OF
            # -------------------------
            atop = on_top_of_score(Ai, Bj, up_axis)
            if atop > 0:
                edges.append(RelationEdge(Ai.class_name, Bj.class_name, "ON_TOP_OF", atop, dist))

            btop = on_top_of_score(Bj, Ai, up_axis)
            if btop > 0:
                edges.append(RelationEdge(Bj.class_name, Ai.class_name, "ON_TOP_OF", btop, dist))

            # -------------------------
            # INSIDE
            # -------------------------
            if inside_score(Ai, Bj) > 0:
                edges.append(RelationEdge(Ai.class_name, Bj.class_name, "INSIDE", 1.0, 0.0))
            if inside_score(Bj, Ai) > 0:
                edges.append(RelationEdge(Bj.class_name, Ai.class_name, "INSIDE", 1.0, 0.0))

    return edges

# ---------------------------------------------------------
# SEMANTIC VLM METHOD
# ---------------------------------------------------------
def draw_bounding_box_and_crop_with_labels(
        full_frame,
        xyxyA, xyxyB,
        labelA="A",
        labelB="B",
        pad=10,
        colorA=(255, 0, 0),     # red
        colorB=(0, 255, 0),     # green
        thickness=3,
        font_scale=0.8,
        font_thickness=2):
    """
    full_frame : np.ndarray (H,W,3) RGB or BGR
    xyxyA, xyxyB : [x1,y1,x2,y2] integer bounding boxes
    labelA, labelB : strings (e.g., object.class_name)
    Returns:
        cropped image with boxes + labels drawn.
    """
    # global crop
    x1 = min(xyxyA[0], xyxyB[0]) - pad
    y1 = min(xyxyA[1], xyxyB[1]) - pad
    x2 = max(xyxyA[2], xyxyB[2]) + pad
    y2 = max(xyxyA[3], xyxyB[3]) + pad

    H, W = full_frame.shape[:2]
    x1 = max(0, x1) 
    y1 = max(0, y1)
    x2 = min(W, x2)
    y2 = min(H, y2)

    crop = full_frame[y1:y2, x1:x2].copy()

    Ax1, Ay1, Ax2, Ay2 = xyxyA
    Bx1, By1, Bx2, By2 = xyxyB

    Ax1 -= x1; Ax2 -= x1
    Ay1 -= y1; Ay2 -= y1

    Bx1 -= x1; Bx2 -= x1
    By1 -= y1; By2 -= y1

    # draw boxes
    cv2.rectangle(crop, (Ax1, Ay1), (Ax2, Ay2), colorA, thickness)
    cv2.rectangle(crop, (Bx1, By1), (Bx2, By2), colorB, thickness)

    # draw labels
    font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_label(img, text, x1, y1, color):
        # Compute text size
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        # Position text — slightly above box
        text_x = x1
        text_y = max(y1 - 5, th + 5)   # avoid going above crop

        # Background rectangle for readability
        cv2.rectangle(img,
                      (text_x, text_y - th - 4),
                      (text_x + tw + 4, text_y + 4),
                      (0, 0, 0),
                      -1)

        # Text
        cv2.putText(img,
                    text,
                    (text_x + 2, text_y - 2),
                    font,
                    font_scale,
                    color,
                    font_thickness,
                    lineType=cv2.LINE_AA)

    # Draw labels for A and B
    draw_label(crop, labelA, Ax1, Ay1, colorA)
    draw_label(crop, labelB, Bx1, By1, colorB)

    return crop

def arbitrate_relation(A, B, votes, llm_model):
    relation_list = "\n".join(f"- {v}" for v in votes)

    prompt = f"""
You are given multiple candidate spatial relation phrases describing how 
object A is positioned relative to object B.

These phrases come from independent vision-language model calls. 
Some of them will be noisy, contradictory, or outliers.

Your task:
1. Identify the underlying relation that is most consistent across the options.
2. Ignore outliers and overly specific or hallucinated phrases.
3. Choose a final relation that is:
   - short (MAX 5 WORDS)
   - purely spatial
   - a noun/adjective/preposition phrase, NOT a full sentence
   - faithful to the majority semantic meaning, not the majority wording

If you believe there is NO meaningful spatial relation, output exactly:
    "no relation"

Return ONLY the final relation phrase.

----
Object A: {A}
Object B: {B}

Candidate relation phrases:
{relation_list}
----
Final relation (MAX 5 WORDS):
"""

    return llm_model(prompt).strip().lower()


# ---------------------------------------------------------
# Main: VLM-based Relation Extraction
# ---------------------------------------------------------
def compute_vlm_relations(objects, frame_loader, vlm_model, llm_model, top_k=5):
    """
    objects: list[MapObject]
        Each MapObject has .crops which is a list of:
            (crop, xyxy, frame_idx, score)
    frame_loader: function(frame_idx) -> RGB frame (numpy array)
        You supply a function that loads the ORIGINAL RGB frame.
    vlm_model: a callable (prompt, image) -> string
    top_k: number of best co-visible frames per object pair

    RETURNS:
        list[RelationEdge]
    """

    # ---------------------------------------------
    # Build per-pair candidate frame lists
    # ---------------------------------------------
    pair_to_frames = defaultdict(list)

    # Produce a small index of crops per object, grouped by frame
    obj_frames = defaultdict(lambda: defaultdict(list))
    for obj_id, obj in enumerate(objects):
        for crop, xyxy, fidx, score in obj.crops:
            obj_frames[obj_id][fidx].append((xyxy, score))

    # For every frame, find objects that appear together
    for objA in range(len(objects)):
        for objB in range(objA+1, len(objects)):
            # Get common frames
            common_frames = set(obj_frames[objA].keys()) & set(obj_frames[objB].keys())
            if not common_frames:
                continue

            # Score pairs by visibility quality (sum of crop scores)
            scored = []
            for f in common_frames:
                scoreA = max(s for (xy,s) in obj_frames[objA][f])
                scoreB = max(s for (xy,s) in obj_frames[objB][f])
                scored.append((f, scoreA + scoreB))

            scored.sort(key=lambda x: x[1], reverse=True)
            pair_to_frames[(objA, objB)] = scored[:top_k]

    # ---------------------------------------------
    # Run VLM on best frames
    # ---------------------------------------------
    final_edges = []

    for (objA, objB), frames in pair_to_frames.items():
        votes = []

        for fidx, _ in frames:
            frame = frame_loader(fidx)             # full RGB frame

            # Get best crops in this frame
            xyxyA = sorted(obj_frames[objA][fidx], key=lambda x: x[1], reverse=True)[0][0]
            xyxyB = sorted(obj_frames[objB][fidx], key=lambda x: x[1], reverse=True)[0][0]

            # Build pair crop
            pair_img = draw_bounding_box_and_crop_with_labels(frame, xyxyA, xyxyB, labelA=objects[objA].class_name, labelB=objects[objB].class_name, pad=10)

            # Infer relation
            pred = vlm_model(
            f"""
You are given an image containing EXACTLY TWO objects.
Each object is clearly labeled with a bounding box and a text label.

Your task:
Fill in the blank in a spatial relation phrase:

    "The [{objects[objA].class_name}] is ___ the [{objects[objB].class_name}]."

Rules for the blank:
- MUST be a spatial relation phrase (1–4 words)
- MUST be directional relative to the SECOND object
- MUST NOT be a full sentence
- MUST NOT include verbs (e.g., "is", "looks", "seems")
- MUST NOT hallucinate new objects or details
- If the spatial relation is unclear or meaningless, answer exactly:
        "no meaningful relation"

Allowed types of phrases:
- relative position (e.g., "in front of", "behind", "next to", "to the left of")
- containment (e.g., "inside", "on top of", "under")
- distance relations (e.g., "far from", "near")
- open set is allowed, but keep it spatial only.

Return ONLY the phrase for the blank.
"""
            , image=pair_img).strip().lower()

            votes.append(pred)
            print(f"Frame {fidx}: [{objects[objA].class_name}] is [{pred}] [{objects[objB].class_name}]")

        if votes:
            final = arbitrate_relation(objects[objA].class_name,
                                    objects[objB].class_name,
                                    votes,
                                    llm_model)

            if final != "no relation":
                final_edges.append(RelationEdge(
                    src_id=objects[objA].oid,
                    dist_id=objects[objB].oid,
                    src=objects[objA].class_name,
                    dst=objects[objB].class_name,
                    rtype=final,
                    score=1.0,
                    dist=0.0
                ))


    return final_edges

# -----------------------------------------
# Save to disk
# -----------------------------------------
def save_relations(edges, path):
    data = [edge.__dict__ for edge in edges]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


