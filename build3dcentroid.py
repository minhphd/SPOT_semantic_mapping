import os
import json
from glob import glob
import cv2
import numpy as np


# ============================================================
# Config
# ============================================================
ROOT = "./2ebf66b16d"

RGB_DIR = os.path.join(ROOT, "rgb_frames")
TRACKS_JSON = os.path.join(ROOT, "fastsam_tracks.json")
OUTPUT_VIDEO = os.path.join(ROOT, "tracked_objects.mp4")

FRAME_SKIP = 2    # draw only every 2nd frame (speed-up)
FPS = 30.0        # output video framerate


# ============================================================
# Load tracking results
# ============================================================
with open(TRACKS_JSON, "r") as f:
    tracks_data = json.load(f)

# mapping: frame_index → objects for that frame
frame_to_objs = {rec["frame_index"]: rec["objects"] for rec in tracks_data}


# deterministic color per object ID
def id_to_color(obj_id):
    np.random.seed(obj_id)
    return (np.random.rand(3) * 255).astype(np.uint8)


# ============================================================
# Render video
# ============================================================
def main():
    rgb_files = sorted(glob(os.path.join(RGB_DIR, "*.jpg")))
    if len(rgb_files) == 0:
        print("[ERROR] No RGB frames found.")
        return

    # determine frame dimensions
    sample = cv2.imread(rgb_files[0], cv2.IMREAD_COLOR)
    H, W = sample.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (W, H))

    print(f"[INFO] Starting video rendering ({len(rgb_files)} frames)...")

    for frame_idx, img_path in enumerate(rgb_files):
        if frame_idx % FRAME_SKIP != 0:
            continue

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Cannot read {img_path}, skipping.")
            continue

        objs = frame_to_objs.get(frame_idx, [])

        for obj in objs:
            obj_id = obj["id"]
            x1, y1, x2, y2 = map(int, obj["bbox"])

            col = id_to_color(obj_id)
            col_bgr = (int(col[2]), int(col[1]), int(col[0]))

            # draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), col_bgr, 2)

            # label
            cv2.putText(
                img,
                f"ID {obj_id}",
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                col_bgr,
                2,
                cv2.LINE_AA,
            )

        writer.write(img)

        print(f"[INFO] Rendered frame {frame_idx+1}/{len(rgb_files)}", end="\r")

    writer.release()
    print(f"\n[INFO] Video saved to: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
