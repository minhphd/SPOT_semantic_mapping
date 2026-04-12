"""
viz.py
======
Visualization utilities for the localization comparison experiment:
  - plot_heatmap: scatter of predicted vs ground-truth positions,
    optionally overlaid on a floorplan image.
  - save_results_csv: write the aggregated results table to CSV.
"""

import re
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path


# ----------------------------------------------------------------------
# Optional: parse room rectangles from mapping.txt for floorplan overlay
# ----------------------------------------------------------------------

def _parse_rooms_for_viz(mapping_txt_path: str) -> list[dict]:
    """Return list of room dicts with label + (x1,y1,x2,y2) in world coords."""
    rooms = []
    label_re = re.compile(r'label="([^"]+)"')
    rect_re  = re.compile(r'rect="([^"]+)"')
    with open(mapping_txt_path) as f:
        for line in f:
            if not line.strip().startswith("ADD_ROOM"):
                continue
            lm = label_re.search(line)
            rm = rect_re.search(line)
            if not (lm and rm):
                continue
            coords = [float(v) for v in rm.group(1).split(",")]
            if len(coords) != 4:
                continue
            x1, y1, x2, y2 = coords
            rooms.append({"label": lm.group(1),
                          "x1": min(x1, x2), "y1": min(y1, y2),
                          "x2": max(x1, x2), "y2": max(y1, y2)})
    return rooms


def _world_to_pixel(x, y, x_min, y_min, x_max, y_max, img_w, img_h):
    """Map world (x, y) to image pixel (px, py)."""
    px = (x - x_min) / (x_max - x_min) * img_w
    py = (1.0 - (y - y_min) / (y_max - y_min)) * img_h  # flip y for image coords
    return px, py


# ----------------------------------------------------------------------
# Main heatmap function
# ----------------------------------------------------------------------

def plot_heatmap(
    method_name: str,
    pred_positions: np.ndarray,
    gt_positions: np.ndarray,
    errors_m: np.ndarray,
    floorplan_path: str | None = None,
    mapping_txt: str | None = None,
    output_path: str = "heatmap.png",
    error_clip_m: float = 10.0,
):
    """
    Plot predicted positions colour-coded by localisation error.

    Parameters
    ----------
    method_name    : name shown in plot title
    pred_positions : (T, 3) predicted XYZ
    gt_positions   : (T, 3) ground-truth XYZ
    errors_m       : (T,)   per-timestep 2D error in metres
    floorplan_path : path to floorplan.png (optional)
    mapping_txt    : path to mapping.txt — used to derive world-coord extents
                     for the floorplan overlay (required if floorplan_path set)
    output_path    : where to save the PNG
    error_clip_m   : errors above this value are clipped for colouring
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    clipped_err = np.clip(errors_m, 0, error_clip_m)
    norm = plt.Normalize(vmin=0, vmax=error_clip_m)
    cmap = cm.viridis

    if floorplan_path is not None and Path(floorplan_path).exists():
        img = plt.imread(floorplan_path)
        img_h, img_w = img.shape[:2]

        # Derive world extents from mapping.txt rooms
        if mapping_txt and Path(mapping_txt).exists():
            rooms = _parse_rooms_for_viz(mapping_txt)
            x_min = min(r["x1"] for r in rooms)
            x_max = max(r["x2"] for r in rooms)
            y_min = min(r["y1"] for r in rooms)
            y_max = max(r["y2"] for r in rooms)
        else:
            # Fallback: use trajectory extents
            all_x = np.concatenate([pred_positions[:, 0], gt_positions[:, 0]])
            all_y = np.concatenate([pred_positions[:, 1], gt_positions[:, 1]])
            margin = 2.0
            x_min, x_max = all_x.min() - margin, all_x.max() + margin
            y_min, y_max = all_y.min() - margin, all_y.max() + margin

        ax.imshow(img, extent=[x_min, x_max, y_min, y_max],
                  aspect="auto", origin="upper", alpha=0.4, cmap="gray")

        # Ground truth trajectory
        ax.plot(gt_positions[:, 0], gt_positions[:, 1],
                color="cyan", linewidth=1.0, alpha=0.6, label="GT trajectory")

        # Predicted positions coloured by error
        sc = ax.scatter(pred_positions[:, 0], pred_positions[:, 1],
                        c=clipped_err, cmap=cmap, norm=norm,
                        s=20, zorder=5, label="Predictions")

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    else:
        # Plain 2D scatter without floorplan
        ax.plot(gt_positions[:, 0], gt_positions[:, 1],
                color="cyan", linewidth=1.0, alpha=0.6, label="GT trajectory")
        sc = ax.scatter(pred_positions[:, 0], pred_positions[:, 1],
                        c=clipped_err, cmap=cmap, norm=norm,
                        s=20, zorder=5, label="Predictions")

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(f"2D error (m), clipped at {error_clip_m}m")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Localisation heatmap — {method_name}")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Saved heatmap → {output_path}")


# ----------------------------------------------------------------------
# Save results to CSV
# ----------------------------------------------------------------------

def save_results_csv(results: dict[str, dict], output_path: str):
    """
    Write aggregated results to a CSV file.

    Parameters
    ----------
    results     : {method_name: aggregate_dict from exp_metrics.aggregate_results}
    output_path : path for the CSV file
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["method", "N",
                  "r@1", "r@3", "r@5", "r@10",
                  "acc_3m", "acc_5m", "room_acc",
                  "mean_err_m", "median_err_m", "room_acc_N"]

    def _fmt(v):
        return f"{v:.2f}" if not np.isnan(v) else "nan"

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method, r in results.items():
            writer.writerow({
                "method":       method,
                "N":            r["N"],
                "r@1":          _fmt(r.get("r@1",  float("nan"))),
                "r@3":          _fmt(r.get("r@3",  float("nan"))),
                "r@5":          _fmt(r.get("r@5",  float("nan"))),
                "r@10":         _fmt(r.get("r@10", float("nan"))),
                "acc_3m":       f"{r['acc_3m']:.2f}",
                "acc_5m":       f"{r['acc_5m']:.2f}",
                "room_acc":     _fmt(r["room_acc"]),
                "mean_err_m":   f"{r['mean_err_m']:.3f}",
                "median_err_m": f"{r['median_err_m']:.3f}",
                "room_acc_N":   r["room_acc_N"],
            })

    print(f"[viz] Saved results CSV → {output_path}")
