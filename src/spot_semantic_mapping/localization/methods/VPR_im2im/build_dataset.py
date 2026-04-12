"""
build_dataset.py
================
Builds VPR evaluation datasets at one or more ground-truth distance thresholds
(e.g. 3 m and 5 m) and automatically attaches room labels to every DB and query
position by testing against the rectangular room boundaries defined in
mapping.txt.

Usage
-----
python build_dataset.py \
  [--spot_ds_path dataset/spot/millerst/data] \
  [--db_path      dataset/3578aa5730] \
  [--mapping_txt  ./mapping.txt] \
  [--output_dir   src/spot_semantic_mapping/localization/methods/VPR_im2im/dataset/] \
  [--windows 3.0 5.0] \
  [--step 10]

Output
------
For each window size W in --windows, saves:
  {output_dir}/spot_dataset_w_gt_{W:.0f}m.pkl

Each pickle contains the standard load_spot_data() dict plus two new fields:
  dataset["db_room_labels"]    — list[str], one label per DB frame
  dataset["query_room_labels"] — list[str], one label per query timestep t
"""

import argparse
import os
import re
import pickle as pkl

import numpy as np

from spot_semantic_mapping.localization.methods.VPR_im2im.make_dataset import load_spot_data


# ----------------------------------------------------------------------
# mapping.txt parser
# ----------------------------------------------------------------------

def parse_rooms(mapping_txt_path):
    """
    Parse ADD_ROOM lines from mapping.txt.

    Returns
    -------
    list of dicts:
        {"label": str, "x1": float, "y1": float, "x2": float, "y2": float}
    where (x1, y1) and (x2, y2) are the min and max corners of the rectangle
    in the Spot odom XY ground plane.
    """
    rooms = []
    label_re = re.compile(r'label="([^"]+)"')
    rect_re  = re.compile(r'rect="([^"]+)"')

    with open(mapping_txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("ADD_ROOM"):
                continue

            label_m = label_re.search(line)
            rect_m  = rect_re.search(line)
            if not (label_m and rect_m):
                print(f"[WARN] Skipping malformed ADD_ROOM line: {line}")
                continue

            label = label_m.group(1)
            coords = [float(v) for v in rect_m.group(1).split(",")]
            if len(coords) != 4:
                print(f"[WARN] Expected 4 rect values, got {len(coords)}: {line}")
                continue

            x1, y1, x2, y2 = coords
            # Normalise so x1 <= x2 and y1 <= y2
            rooms.append({
                "label": label,
                "x1": min(x1, x2),
                "y1": min(y1, y2),
                "x2": max(x1, x2),
                "y2": max(y1, y2),
            })

    print(f"Parsed {len(rooms)} room(s) from {mapping_txt_path}:")
    for r in rooms:
        print(f"  {r['label']:30s}  x=[{r['x1']:.2f}, {r['x2']:.2f}]  y=[{r['y1']:.2f}, {r['y2']:.2f}]")
    return rooms


# ----------------------------------------------------------------------
# Room label assignment
# ----------------------------------------------------------------------

def label_positions(traj, rooms):
    """
    Assign a room label to each position in traj.

    Parameters
    ----------
    traj  : np.ndarray, shape (N, 3)  — columns are (x, y, z) in Spot odom frame
    rooms : list of dicts from parse_rooms()

    Returns
    -------
    list[str] of length N — room label or "unknown" if no room matches
    """
    labels = []
    xs = traj[:, 0]
    ys = traj[:, 1]

    for x, y in zip(xs, ys):
        assigned = "unknown"
        for r in rooms:
            if r["x1"] <= x <= r["x2"] and r["y1"] <= y <= r["y2"]:
                assigned = r["label"]
                break
        labels.append(assigned)

    return labels


# ----------------------------------------------------------------------
# Dataset builder
# ----------------------------------------------------------------------

def build_datasets(spot_ds_path, db_path, mapping_txt, output_dir, windows, step):
    """
    Build and save one pickle per window size.
    """
    rooms = parse_rooms(mapping_txt)
    os.makedirs(output_dir, exist_ok=True)

    for window_size in windows:
        print(f"\n{'='*60}")
        print(f"Building dataset  window_size={window_size} m")
        print(f"{'='*60}")

        dataset = load_spot_data(
            spot_ds_path=spot_ds_path,
            db_path=db_path,
            step=step,
            window_size=window_size,
        )

        # Attach room labels
        if dataset["db_traj"] is not None and len(dataset["db_traj"]) > 0:
            dataset["db_room_labels"] = label_positions(dataset["db_traj"], rooms)
        else:
            print("[WARN] db_traj is empty — db_room_labels will be empty list")
            dataset["db_room_labels"] = []

        if dataset["query_traj"] is not None and len(dataset["query_traj"]) > 0:
            dataset["query_room_labels"] = label_positions(dataset["query_traj"], rooms)
        else:
            print("[WARN] query_traj is empty — query_room_labels will be empty list")
            dataset["query_room_labels"] = []

        # Summary
        from collections import Counter
        db_counts    = Counter(dataset["db_room_labels"])
        query_counts = Counter(dataset["query_room_labels"])
        print(f"\nDB    room label distribution : {dict(db_counts)}")
        print(f"Query room label distribution : {dict(query_counts)}")

        tag = f"{window_size:.0f}m"
        out_path = os.path.join(output_dir, f"spot_dataset_w_gt_{tag}.pkl")
        with open(out_path, "wb") as f:
            pkl.dump(dataset, f)
        print(f"\nSaved → {out_path}")

    print("\nDone.")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    default_output = os.path.join(
        os.path.dirname(__file__), "dataset"
    )

    parser = argparse.ArgumentParser(
        description="Build Spot VPR evaluation datasets with room labels."
    )
    parser.add_argument(
        "--spot_ds_path",
        default="dataset/spot/millerst/data",
        help="Path to the query SpotDataset directory (default: %(default)s)",
    )
    parser.add_argument(
        "--db_path",
        default="dataset/3578aa5730",
        help="Path to the DB traversal directory (default: %(default)s)",
    )
    parser.add_argument(
        "--mapping_txt",
        default="./mapping.txt",
        help="Path to mapping.txt containing ADD_ROOM definitions (default: %(default)s)",
    )
    parser.add_argument(
        "--output_dir",
        default=default_output,
        help="Directory to write output pkl files (default: %(default)s)",
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        type=float,
        default=[3.0, 5.0],
        help="Ground-truth distance threshold(s) in metres (default: 3.0 5.0)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=10,
        help="Subsampling step for DB frames (default: %(default)s)",
    )

    args = parser.parse_args()

    build_datasets(
        spot_ds_path=args.spot_ds_path,
        db_path=args.db_path,
        mapping_txt=args.mapping_txt,
        output_dir=args.output_dir,
        windows=args.windows,
        step=args.step,
    )


if __name__ == "__main__":
    main()
