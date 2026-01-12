import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d
import faiss
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ============================================================
# Utility
# ============================================================

def l2_normalize(v, eps=1e-6):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v if n < eps else v / n


# ============================================================
# Detection = one object instance in one frame
# ============================================================

@dataclass
class Detection:
    """Single-frame detected object (ConceptGraph style)."""
    pcd: o3d.geometry.PointCloud
    bbox: o3d.geometry.AxisAlignedBoundingBox
    clip_ft: torch.Tensor
    crop: Optional[tuple] = field(default_factory=lambda: (None, None, None, None))  # (crop, xyxy, frame_idx, score)
    # xyxy: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    class_id: Optional[int] = None
    class_name: Optional[str] = None
    # frame_idx: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def set_crop(self, crop: tuple):
        cut, xyxy, frame_idx = crop
        h, w = cut.shape[:2]
        area = h * w
        # variance: indicator of detail/structure
        variance = float(np.var(cut))
        score = area + 0.5 * variance  # combine metrics
        self.crop = (cut, xyxy, frame_idx, score)

class DetectionList(list):
    """List of detections with helper stacking."""
    def get_stacked_values_torch(self, key: str) -> torch.Tensor:
        vals = []
        for det in self:
            v = getattr(det, key)
            if not isinstance(v, torch.Tensor):
                v = torch.as_tensor(v)
            vals.append(v)
        return torch.stack(vals, dim=0)   # (M, D)


# ============================================================
# Persistent Tracked Objects
# ============================================================

@dataclass
class MapObject:
    """Multi-frame aggregated 3D object."""
    pcd: o3d.geometry.PointCloud
    bbox: o3d.geometry.AxisAlignedBoundingBox
    clip_ft: torch.Tensor                   # aggregated feature
    crops: List[tuple] = field(default_factory=list)  # list of (crop, xyxy, frame_idx, score)
    oid: Optional[int] = None
    # xyxys: Optional[List[np.ndarray]] = field(default_factory=list)
    class_ids: List[int] = field(default_factory=list)
    class_name: str = ""
    detections: List[Detection] = field(default_factory=list)
    num_views: int = 1
    total_weight: float = 1.0
    
    def sort_crops(self):
        """Sort crops by score descending."""
        sorted_indices = np.argsort([crop[3] for crop in self.crops])[::-1]
        self.crops = [self.crops[i] for i in sorted_indices]

class MapObjectList(list):
    def get_stacked_values_torch(self, key: str) -> torch.Tensor:
        vals = []
        for obj in self:
            v = getattr(obj, key)
            if not isinstance(v, torch.Tensor):
                v = torch.as_tensor(v)
            vals.append(v)
        return torch.stack(vals, dim=0)


# ============================================================
# Geometry: Overlap (FAISS)
# ============================================================

def compute_overlap_diagonal(detections, objects, voxel_size):
    """
    Compute spatial overlap ONLY for pairs (det[i], obj[i]).
    Assumes len(detections) == len(objects).
    """
    K = len(detections)
    out = np.zeros(K, dtype=np.float32)

    for i in range(K):
        det_pts = np.asarray(detections[i].pcd.points, dtype=np.float32)
        obj_pts = np.asarray(objects[i].pcd.points, dtype=np.float32)

        if len(det_pts) == 0 or len(obj_pts) == 0:
            continue

        # FAISS index for this object
        idx = faiss.IndexFlatL2(obj_pts.shape[1])
        idx.add(obj_pts)

        D, I = idx.search(det_pts, 1)
        matched = (D < voxel_size ** 2).sum()
        out[i] = matched / len(det_pts)

    return torch.tensor(out, dtype=torch.float32)

def compute_overlap_diagonal_gpu(
    det_subset: DetectionList,
    obj_subset: MapObjectList,
    voxel_size: float,
    device="cuda",
    batch_size=20000):
    """
    Efficient diagonal overlap:
    For each pair (det_i, obj_i), compute:
        fraction of det_i points with NN < voxel_size from obj_i

    Batched so GPU memory does not creep even on huge point clouds.
    """
    K = len(det_subset)
    out = torch.zeros(K, device=device)

    for idx in range(K):
        det_pts = torch.tensor(
            np.asarray(det_subset[idx].pcd.points), device=device, dtype=torch.float32
        )
        obj_pts = torch.tensor(
            np.asarray(obj_subset[idx].pcd.points), device=device, dtype=torch.float32
        )

        if det_pts.shape[0] == 0 or obj_pts.shape[0] == 0:
            continue

        # Incremental computation to avoid GPU memory creep
        nearest = []

        for start in range(0, det_pts.shape[0], batch_size):
            end = min(start + batch_size, det_pts.shape[0])
            det_chunk = det_pts[start:end]

            # GPU pairwise distances
            dist = torch.cdist(det_chunk, obj_pts, p=2)  # (chunk, obj_pts)
            min_d = torch.min(dist, dim=1)[0]
            nearest.append(min_d)

            del dist, min_d, det_chunk  # prevent memory creep
            torch.cuda.empty_cache()

        nearest = torch.cat(nearest)
        out[idx] = (nearest < voxel_size).float().mean()

    return out

# ============================================================
# Visual Similarity (CLIP cosine)
# ============================================================

def compute_visual_similarities(detections: DetectionList,
                                objects: MapObjectList) -> torch.Tensor:
    if len(detections) == 0 or len(objects) == 0:
        return torch.zeros((len(detections), len(objects)))

    det_fts = detections.get_stacked_values_torch('clip_ft')  # (M, D)
    obj_fts = objects.get_stacked_values_torch('clip_ft')     # (N, D)

    det_norm = F.normalize(det_fts, dim=1)
    obj_norm = F.normalize(obj_fts, dim=1)

    return det_norm @ obj_norm.T   # (M, N)


# ============================================================
# Aggregation: spatial + visual
# ============================================================

def aggregate_similarities(spatial_sim: torch.Tensor,
                           visual_sim: torch.Tensor,
                           w_geo=1.0,
                           w_sem=1.0) -> torch.Tensor:
    return w_geo * spatial_sim + w_sem * visual_sim


# ============================================================
# Matching detections → objects
# ============================================================

def match_detections_to_objects(agg_sim: torch.Tensor,
                                threshold: float = 0.3) -> List[Optional[int]]:
    match_indices = []
    for i in range(agg_sim.shape[0]):
        row = agg_sim[i]
        j = torch.argmax(row).item()
        if row[j] < threshold:
            match_indices.append(None)
        else:
            match_indices.append(j)
    return match_indices


# ============================================================
# Merging detections into objects
# ============================================================

def merge_detection_into_object(obj: MapObject, det: Detection,
                                voxel_size=0.01,
                                max_points=20000) -> MapObject:

    # Merge PCD
    merged_pcd = obj.pcd + det.pcd
    merged_pcd = merged_pcd.voxel_down_sample(voxel_size)
    merged_bbox = merged_pcd.get_axis_aligned_bounding_box()
    if len(merged_pcd.points) > max_points:
        merged_pcd = merged_pcd.random_down_sample(
            max_points / len(merged_pcd.points)
        )

    # -----------------------------
    # Update CLIP feature (average by number of points)
    # -----------------------------
    obj_num_pts = len(obj.pcd.points)
    det_num_pts = len(det.pcd.points)
    total_pts = obj_num_pts + det_num_pts

    new_feat = (obj.clip_ft * obj_num_pts + det.clip_ft * det_num_pts) / total_pts
    obj.clip_ft = F.normalize(new_feat, dim=0)

    # Update fields
    obj.pcd = merged_pcd
    obj.bbox = merged_bbox
    obj.crops.append(det.crop)
    obj.detections.append(det)
    obj.num_views += 1

    if det.class_id is not None:
        obj.class_ids.append(det.class_id)

    return obj



# ============================================================
# Main tracker class
# ============================================================

class ObjectTracker3D:
    """
    ConceptGraph-style 3D tracker but with critical speed-ups:
        - spatial + visual similarity computed ONLY for nearby objects
        - far objects are masked out entirely (never matched)
    """

    def __init__(self,
                 voxel_size=0.01,
                 w_geo=1.0,
                 w_sem=1.0,
                 match_threshold=0.3,
                 center_dist_thresh=0.7,      # <---- NEW
                 class_gate=True,
                 max_points=20000,
                 edges = []):              # <---- NEW
        self.objects = MapObjectList()
        self.voxel_size = voxel_size
        self.w_geo = w_geo
        self.w_sem = w_sem
        self.edges = edges
        self.match_threshold = match_threshold
        self.max_points = max_points
        
        # gating
        self.center_dist_thresh = center_dist_thresh
        self.class_gate = class_gate

    # --------------------------------------------------------------
    # Utility
    # --------------------------------------------------------------
    def _bbox_center(self, bbox: o3d.geometry.AxisAlignedBoundingBox):
        return np.asarray(bbox.get_center(), dtype=np.float32)

    # --------------------------------------------------------------
    # Main update
    # --------------------------------------------------------------
    def update(self, detection_list: DetectionList):

        M = len(detection_list)
        if M == 0:
            return

        # ----------------------------------------------------------
        # Case 1 — No previous objects → initialize
        # ----------------------------------------------------------
        if len(self.objects) == 0:
            for det in detection_list:
                self.objects.append(
                    MapObject(
                        oid = len(self.objects),
                        pcd=det.pcd,
                        bbox=det.bbox,
                        clip_ft=det.clip_ft.clone(),
                        crops=[det.crop],
                        class_ids=[det.class_id] if det.class_id is not None else [],
                        class_name=det.class_name or "",
                        detections=[det],
                    )
                )
            return

        # ----------------------------------------------------------
        # Precompute geometric info
        # ----------------------------------------------------------
        det_centers = np.stack([self._bbox_center(d.bbox) for d in detection_list])  # (M, 3)
        obj_centers = np.stack([self._bbox_center(o.bbox) for o in self.objects])     # (N, 3)

        det_class = np.array([d.class_id for d in detection_list])
        obj_class = np.array([o.class_ids[-1] if o.class_ids else -1 for o in self.objects])

        # ----------------------------------------------------------
        # Compute distance matrix (M × N)
        # ----------------------------------------------------------
        diff = det_centers[:, None, :] - obj_centers[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)  # (M, N)

        # gating: mark nearby objects
        gating_mask = (dist < self.center_dist_thresh)

        # optional class gating
        if self.class_gate:
            class_mask = det_class[:, None] == obj_class[None, :]
            gating_mask = gating_mask & class_mask

        # ----------------------------------------------------------
        # Spatial similarity ONLY for gated pairs
        # ----------------------------------------------------------
        spatial_sim = torch.zeros((M, len(self.objects)), dtype=torch.float32, device="cpu")
        gated_det_idx, gated_obj_idx = np.where(gating_mask)
        gated_pairs = list(zip(gated_det_idx, gated_obj_idx))
        if len(gated_pairs) > 0:

            # Build subsets in detection order
            det_subset = DetectionList([detection_list[i] for i, _ in gated_pairs])
            obj_subset = MapObjectList([self.objects[j] for _, j in gated_pairs])

            partial_spatial = compute_overlap_diagonal_gpu(
                det_subset, obj_subset, self.voxel_size, device="cuda"
            ).cpu()

            for k, (i, j) in enumerate(gated_pairs):
                spatial_sim[i, j] = partial_spatial[k]

        # ----------------------------------------------------------
        # Visual similarity ONLY for gated pairs
        # ----------------------------------------------------------
        visual_sim = torch.zeros((M, len(self.objects)), dtype=torch.float32)

        if len(gated_det_idx) > 0:
            det_fts = detection_list.get_stacked_values_torch("clip_ft")
            obj_fts = self.objects.get_stacked_values_torch("clip_ft")

            det_norm = F.normalize(det_fts, dim=1)
            obj_norm = F.normalize(obj_fts, dim=1)

            full_visual = det_norm @ obj_norm.T  # (M, N)

            visual_sim[gated_det_idx, gated_obj_idx] = full_visual[gated_det_idx, gated_obj_idx]

        # ----------------------------------------------------------
        # Combine similarities
        # ----------------------------------------------------------
        agg_sim = self.w_geo * spatial_sim + self.w_sem * visual_sim
        # ----------------------------------------------------------
        # Perform matching
        # ----------------------------------------------------------
        match_indices = match_detections_to_objects(
            agg_sim, threshold=self.match_threshold
        )

        # ----------------------------------------------------------
        # Merge detections
        # ----------------------------------------------------------
        for det_idx, obj_idx in enumerate(match_indices):
            det = detection_list[det_idx]

            if obj_idx is None:
                # Create new object
                self.objects.append(
                    MapObject(
                        pcd=det.pcd,
                        oid=len(self.objects),
                        bbox=det.bbox,
                        clip_ft=det.clip_ft.clone(),
                        crops=[det.crop],
                        class_ids=[det.class_id] if det.class_id else [],
                        class_name=det.class_name or "",
                        detections=[det],
                    )
                )
            else:
                # Merge into existing
                obj = self.objects[obj_idx]
                merged = merge_detection_into_object(obj, det, voxel_size=self.voxel_size, max_points=self.max_points)
                self.objects[obj_idx] = merged