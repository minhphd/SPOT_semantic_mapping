"""
exp_metrics.py
==============
Position-based and room-based localization metrics for the paper experiment.
These go beyond retrieval rank metrics to navigation-relevant quantities:
position error (m), 3m / 5m accuracy, room accuracy, and R@K recall.
"""

import numpy as np


# ----------------------------------------------------------------------
# Position prediction from retrieval scores (im2im only)
# ----------------------------------------------------------------------

def predict_position(scores: np.ndarray, db_traj: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Weighted average of the top-k database positions.

    Parameters
    ----------
    scores   : (N_db,) float — similarity / score for each DB frame (higher = better)
    db_traj  : (N_db, 3)    — XYZ position of each DB frame in odom frame
    k        : int          — number of top candidates to use

    Returns
    -------
    pred_pos : (3,) predicted position in odom frame
    """
    k = min(k, len(scores))
    top_k_idx = np.argsort(-scores)[:k]
    raw_w = scores[top_k_idx]

    # Shift so the minimum weight is 0 before softmax (avoids exp of large negatives)
    raw_w = raw_w - raw_w.max()
    w = np.exp(raw_w)
    w /= w.sum() + 1e-12

    return (w[:, None] * db_traj[top_k_idx]).sum(axis=0)


# ----------------------------------------------------------------------
# Per-timestep metrics
# ----------------------------------------------------------------------

def compute_loc_metrics(
    pred_pos: np.ndarray,
    gt_pos: np.ndarray,
    pred_room: str,
    gt_room: str,
    scores: np.ndarray | None = None,
    gt_set: set | None = None,
) -> dict:
    """
    Compute localization metrics for one query timestep.

    Parameters
    ----------
    pred_pos  : (3,) predicted XYZ position
    gt_pos    : (3,) ground-truth XYZ position
    pred_room : predicted room label (top-1 retrieved frame/node's room)
    gt_room   : ground-truth room label from dataset
    scores    : (N_db,) retrieval scores — required for R@K; None for im2graph/g2g
    gt_set    : set of matching DB frame indices for this timestep (from ground_truth)

    Returns
    -------
    dict with keys:
        err_m    — 2D Euclidean error (x, y only; ignores z/height)
        acc_3m   — 1.0 if err_m ≤ 3.0 else 0.0
        acc_5m   — 1.0 if err_m ≤ 5.0 else 0.0
        room_acc — 1.0 / 0.0 / nan (nan when gt_room == "unknown")
        r@1, r@3, r@5, r@10 — Recall@K (nan when scores/gt_set not provided)
    """
    err = float(np.linalg.norm(pred_pos[:2] - gt_pos[:2]))
    room_acc = (
        np.nan if gt_room == "unknown"
        else float(pred_room == gt_room)
    )

    result = {
        "err_m":    err,
        "acc_3m":   float(err <= 3.0),
        "acc_5m":   float(err <= 5.0),
        "room_acc": room_acc,
        "r@1":  np.nan,
        "r@3":  np.nan,
        "r@5":  np.nan,
        "r@10": np.nan,
    }

    if scores is not None and gt_set:
        ranked = np.argsort(-np.asarray(scores))
        for k, key in zip([1, 3, 5, 10], ["r@1", "r@3", "r@5", "r@10"]):
            result[key] = float(any(int(idx) in gt_set for idx in ranked[:k]))

    return result


# ----------------------------------------------------------------------
# Aggregate per-timestep results → summary
# ----------------------------------------------------------------------

def aggregate_results(per_t: list[dict]) -> dict:
    """
    Aggregate a list of per-timestep metric dicts into summary statistics.

    Returns
    -------
    dict with keys:
        mean_err_m, median_err_m,
        acc_3m (%), acc_5m (%),
        room_acc (%), room_acc_N (count of valid timesteps),
        r@1 (%), r@3 (%), r@5 (%), r@10 (%) — nan if no scores provided,
        N (total timesteps)
    """
    errs     = np.array([r["err_m"]   for r in per_t])
    acc3     = np.array([r["acc_3m"]  for r in per_t])
    acc5     = np.array([r["acc_5m"]  for r in per_t])
    room_raw = np.array([r["room_acc"] for r in per_t], dtype=float)

    valid_room = room_raw[~np.isnan(room_raw)]

    agg = {
        "N":            len(per_t),
        "mean_err_m":   float(np.mean(errs)),
        "median_err_m": float(np.median(errs)),
        "acc_3m":       float(np.mean(acc3) * 100),
        "acc_5m":       float(np.mean(acc5) * 100),
        "room_acc":     float(np.mean(valid_room) * 100) if len(valid_room) > 0 else float("nan"),
        "room_acc_N":   int(len(valid_room)),
    }

    for key in ["r@1", "r@3", "r@5", "r@10"]:
        vals = np.array([r[key] for r in per_t], dtype=float)
        valid = vals[~np.isnan(vals)]
        agg[key]         = float(np.mean(valid) * 100) if len(valid) > 0 else float("nan")
        agg[f"{key}_N"]  = int(len(valid))

    return agg


# ----------------------------------------------------------------------
# Pretty-print a results dict
# ----------------------------------------------------------------------

def format_results_table(results: dict[str, dict]) -> str:
    """
    Format aggregated results as a plain-text table.

    Parameters
    ----------
    results : {method_name: aggregate_dict}
    """
    header = (
        f"{'Method':<25} | {'N':>5} | {'R@1':>6} | {'R@3':>6} | "
        f"{'R@5':>6} | {'R@10':>6} | {'3m%':>7} | {'5m%':>7} | "
        f"{'Rm%':>7} | {'MeanErr':>8} | {'MedErr':>8}"
    )
    sep   = "-" * len(header)
    lines = ["", "=" * len(header), header, sep]

    def _fmt(v, fmt=".1f"):
        return f"{v:{fmt}}" if not np.isnan(v) else "  N/A"

    for method, r in results.items():
        room = _fmt(r["room_acc"])
        lines.append(
            f"{method:<25} | {r['N']:5d} | "
            f"{_fmt(r['r@1']):>6} | {_fmt(r['r@3']):>6} | "
            f"{_fmt(r['r@5']):>6} | {_fmt(r['r@10']):>6} | "
            f"{r['acc_3m']:6.2f}% | {r['acc_5m']:6.2f}% | {room}% | "
            f"{r['mean_err_m']:7.2f}m | {r['median_err_m']:7.2f}m"
        )

    lines.append("=" * len(header) + "\n")
    return "\n".join(lines)
