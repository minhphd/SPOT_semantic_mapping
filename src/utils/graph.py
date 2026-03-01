# Classes/object_node.py
"""
ObjectNode: Final object representation used in the scene graph.

This is separate from TrackedObject, which is a temporary
representation used only during tracking.
"""

from dataclasses import dataclass
from typing import List, Dict
# from .object_node import ObjectNodez
import json
import numpy as np

@dataclass
class ObjectNode:
    """
    A finalized object entry used in the semantic scene graph.

    Attributes
    ----------
    id : int
        Unique object identifier.
    label : str
        Generated caption for the object.
    semantic_class : str
        Open-set class label (e.g., "chair", "monitor").
    centroid : list[float]
        Single 3D centroid of the object (x, y, z).
    views : int
        Number of aggregated views.
    bbox_min : list[float]
        Minimum XYZ of bounding box.
    bbox_max : list[float]
        Maximum XYZ of bounding box.
    """
    id: int
    label: str
    semantic_class: str
    centroid: List[float]
    views: int
    bbox_min: List[float]
    bbox_max: List[float]



def export_scene_graph(path, objects, relations: List[Dict]) -> None:
    """
    Save scene graph to a JSON file.

    Parameters
    ----------
    path : str
        Output path.
    objects : list[ObjectNode]
        All objects in the scene.
    relations : list[dict]
        Relations between objects.
    """
    obj_list = []
    for obj in objects:
        obj_list.append({
            "id": obj.id,
            "label": obj.label,
            "class": obj.semantic_class,
            "centroid": obj.centroid,
            "views": obj.views,
            "bbox_min": obj.bbox_min,
            "bbox_max": obj.bbox_max,
        })

    data = {
        "objects": obj_list,
        "relations": relations,
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


@dataclass
class SceneGraph:
    """
    A semantic scene graph consisting of:
      - Object nodes
      - Spatial relations between them

    Attributes
    ----------
    objects : list[ObjectNode]
        All objects with semantic information.
    relations : list[dict]
        Relations of the form:
        {
           "subject": int,
           "predicate": str,
           "object": int
        }
    """
    objects: List[ObjectNode]
    relations: List[Dict]


def export_hierarchical_scene_graph(
    path,
    tracker,
    object_labels,
    object_semantic_class,
    bboxes,
    relations,
    dataset_path=None
):
    """
    Export a full hierarchical scene graph including:

      - Objects w/ centroid, bbox, class, caption, room_id.
      - Room nodes & which objects belong to each room.
      - Relations (near, supports, inside, connected_via_door).
      - Metadata.

    Parameters
    ----------
    path : str
        Destination JSON path.
    tracker : ObjectTracker
    object_labels : dict oid -> caption string
    object_semantic_class : dict oid -> semantic class name (open-set)
    bboxes : dict oid -> {min, max}
    relations : list of {subject, predicate, object}
    dataset_path : str or None

    Produces
    --------
    A JSON file with structure:

    {
      "rooms": [...],
      "objects": [...],
      "relations": [...]
    }
    """
    # ---------------------------------------------------
    # Build room table
    # ---------------------------------------------------
    rooms_dict = {}   # room_id -> { id, centroid, objects: [] }

    # gather object centroids by room
    room_to_pts = {}

    for oid, obj in tracker.objects.items():
        room_id = obj.get("room_id", "room_global")
        room_to_pts.setdefault(room_id, []).append(obj["centroid"])

    # compute room centroids
    for room_id, pts in room_to_pts.items():
        pts_arr = np.vstack(pts)
        centroid = pts_arr.mean(axis=0).tolist()

        rooms_dict[room_id] = {
            "id": room_id,
            "centroid": centroid,
            "objects": []
        }

    # ---------------------------------------------------
    # Fill object entries
    # ---------------------------------------------------
    objects_out = []

    for oid, obj in tracker.objects.items():
        room_id = obj.get("room_id", "room_global")
        centroid = obj["centroid"].tolist()

        entry = {
            "id": int(oid),
            "room_id": room_id,
            "centroid": centroid,
            "views": int(obj["views"]),
            "caption": object_labels.get(oid, "(no caption)"),
            "class": object_semantic_class.get(oid, "(unknown)")
        }

        if oid in bboxes:
            entry["bbox_min"] = bboxes[oid]["min"].tolist()
            entry["bbox_max"] = bboxes[oid]["max"].tolist()

        objects_out.append(entry)
        rooms_dict[room_id]["objects"].append(int(oid))

    # Convert rooms to list
    rooms_out = list(rooms_dict.values())

    # ---------------------------------------------------
    # Final graph struct
    # ---------------------------------------------------
    graph = {
        "dataset_path": dataset_path,
        "rooms": rooms_out,
        "objects": objects_out,
        "relations": relations
    }

    # ---------------------------------------------------
    # Write to file
    # ---------------------------------------------------
    with open(path, "w") as f:
        json.dump(graph, f, indent=2)

    print(f"[SceneGraph] Saved → {path}")
