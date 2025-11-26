import os
import json
from glob import glob
import cv2
import numpy as np

from fastsam import FastSAM, FastSAMPrompt

# ------------------------------
# Config
# ------------------------------
ROOT = "./dataset/millerst_iphone/3578aa5730"
FRAMES_DIR = os.path.join(ROOT, "rgb_frames")

MODEL_PATH = "sam_ckpt/FastSAM-x.pt"
DEVICE = "cuda"

CONF_THRESH = 0.4
IOU_THRESH = 0.5
SAVE_JSON = True

OUT_JSON = os.path.join(ROOT, "fastsam_tracks.json")


# ------------------------------
# IoU
# ------------------------------
def bbox_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    if inter <= 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return inter / (area1 + area2 - inter)


# ------------------------------
# Simple IoU Tracker
# ------------------------------
class SimpleTracker:
    def __init__(self, iou_thresh=0.5):
        self.iou_thresh = iou_thresh
        self.tracks = []      # [{id, bbox}]
        self.next_id = 0

    def update(self, detections):
        assigned_det = set()
        results = []

        # existing track matching
        for ti, track in enumerate(self.tracks):
            best_iou = 0
            best_di = None

            for di, det in enumerate(detections):
                if di in assigned_det:
                    continue
                iou = bbox_iou(track["bbox"], det["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_di = di

            if best_di is not None and best_iou >= self.iou_thresh:
                det = detections[best_di]
                det["id"] = track["id"]
                results.append(det)
                assigned_det.add(best_di)

        # new detections → new track IDs
        for di, det in enumerate(detections):
            if di in assigned_det:
                continue
            det_id = self.next_id
            self.next_id += 1
            det["id"] = det_id
            results.append(det)

        # update tracker state
        self.tracks = [{"id": det["id"], "bbox": det["bbox"]} for det in results]

        return results


# ------------------------------
# Convert FastSAM annotations → bboxes
# ------------------------------
def masks_to_bboxes(ann):
    """
    ann is a list of 2D torch tensors (H x W)
    """
    dets = []

    for mask in ann:
        # Convert tensor → numpy
        mask_np = mask.cpu().numpy()

        ys, xs = np.where(mask_np > 0)

        if len(xs) == 0:
            continue

        x1, y1 = xs.min(), ys.min()
        x2, y2 = xs.max(), ys.max()

        dets.append({
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "score": 1.0,  # everything_prompt has no scores
            "cls": -1,
        })

    return dets



# ------------------------------
# MAIN
# ------------------------------
def main():
    print("[INFO] Loading FastSAM model...")
    model = FastSAM(MODEL_PATH)

    frame_paths = sorted(glob(os.path.join(FRAMES_DIR, "*.jpg")))
    print(f"[INFO] Found {len(frame_paths)} frames.")

    tracker = SimpleTracker(iou_thresh=IOU_THRESH)

    all_results = []

    for frame_idx, fpath in enumerate(frame_paths):
        # if frame_idx == 10:
        #     break  # DEBUG: process only first 10 frames
        if frame_idx % 2 != 0:
            continue  # process every 2nd frame
        img = cv2.imread(fpath)
        if img is None:
            print(f"[WARN] Cannot read {fpath}")
            continue

        # ---------------------------------------------
        # FastSAM inference using original API
        # ---------------------------------------------
        everything_results = model(
            fpath,
            device=DEVICE,
            retina_masks=True,
            imgsz=1024,
            conf=CONF_THRESH,
            iou=0.9,
        )

        prompt_proc = FastSAMPrompt(fpath, everything_results, device=DEVICE)
        ann = prompt_proc.everything_prompt()  # list of segment masks

        # convert masks → bounding boxes
        dets = masks_to_bboxes(ann)

        # tracking
        tracked = tracker.update(dets)

        # save
        frame_record = {
            "frame_index": frame_idx,
            "frame_path": fpath,
            "objects": [
                {
                    "id": obj["id"],
                    "bbox": obj["bbox"],
                    "score": obj["score"],
                    "cls": obj["cls"],
                }
                for obj in tracked
            ]
        }
        all_results.append(frame_record)

        print(f"[INFO] Frame {frame_idx}: {len(tracked)} objects")

    # save JSON
    if SAVE_JSON:
        with open(OUT_JSON, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"[INFO] Saved tracking results to {OUT_JSON}")


if __name__ == "__main__":
    main()
