import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _parse_predictions_cell(x: object) -> np.ndarray:
    """
    The sliding-window CSV stores an array of per-window unstable probabilities in the
    `predictions` column. Depending on pandas/numpy versions it may look like:
      - "[0.1 0.2 0.3]"
      - "[0.1, 0.2, 0.3]"
      - "0.1 0.2 0.3"
    """
    s = str(x).strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    s = s.replace(",", " ")
    arr = np.fromstring(s, sep=" ")
    return arr


def _ensure_predictions_array(x: object) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return _parse_predictions_cell(x)


def _resolve_sliding_window_csv(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.is_dir():
        raise SystemExit(f"Path does not exist: {path}")

    matches = sorted(path.glob("sliding_window_windowsize*_stride*.csv"))
    if len(matches) == 0:
        raise SystemExit(
            f"No sliding-window CSV found in {path}. Expected a file named like "
            f"`sliding_window_windowsize64_stride8.csv`."
        )
    if len(matches) > 1:
        names = "\n  - " + "\n  - ".join(str(p) for p in matches)
        raise SystemExit(f"Multiple sliding-window CSVs found in {path}:{names}\nPlease pass one CSV explicitly.")
    return matches[0]


def _parse_window_and_stride_from_name(csv_path: Path) -> tuple[Optional[int], Optional[int]]:
    match = re.search(r"windowsize(\d+)_stride(\d+)", csv_path.name)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def _safe_stem(s: str) -> str:
    # Keep filenames portable across OS/filesystems.
    s = s.strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("._-") or "item"


# Skeleton bone connections.
# - `ntu25` matches Kinect v2 / NTU 25-joint layout used in `skeleton_render_frames.py`
# - `h36m17` matches Human3.6M 17-joint layout used in `skeleton_animation.py`
_BONE_CONNECTIONS_NTU25: list[tuple[int, int]] = [
    (0, 1),
    (1, 20),
    (20, 2),
    (2, 3),  # Spine to head
    (20, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 21),
    (7, 22),  # Left arm
    (20, 8),
    (8, 9),
    (9, 10),
    (10, 11),
    (11, 23),
    (11, 24),  # Right arm
    (0, 12),
    (12, 13),
    (13, 14),
    (14, 15),  # Left leg
    (0, 16),
    (16, 17),
    (17, 18),
    (18, 19),  # Right leg
]

_BONE_CONNECTIONS_H36M17: list[tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 4),
    (4, 5),
    (5, 6),
    (0, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (8, 11),
    (11, 12),
    (12, 13),
    (8, 14),
    (14, 15),
    (15, 16),
]

def _normalize_shoulder_width(
    points_xyz: np.ndarray,
    *,
    skeleton_format: str,
    target_width: float = 1.0,
) -> np.ndarray:
    """
    Scale the skeleton so shoulder width equals target_width.
    Returns scaled (J, 3) array.
    """
    pts = np.asarray(points_xyz, dtype=np.float32)
    
    if skeleton_format == "ntu25":
        ls_idx, rs_idx = 4, 8  # left shoulder, right shoulder
    elif skeleton_format == "h36m17":
        ls_idx, rs_idx = 11, 14
    else:
        raise ValueError(f"Unknown skeleton_format: {skeleton_format}")
    
    left_sh = pts[ls_idx]
    right_sh = pts[rs_idx]
    shoulder_vec = right_sh - left_sh
    current_width = float(np.linalg.norm(shoulder_vec))
    
    if current_width == 0.0:
        return pts
    
    scale = target_width / current_width
    # Scale only X axis
    pts_scaled = pts.copy()
    pts_scaled[:, 0] *= scale
    
    return pts_scaled

def _apply_isometric_projection(points_xyz: np.ndarray) -> np.ndarray:
    """Apply isometric projection to 3D points. Returns (J, 2)."""
    angle_y = 0
    angle_x = 0

    ry = np.array([
        [np.cos(angle_y), 0, np.sin(angle_y)],
        [0, 1, 0],
        [-np.sin(angle_y), 0, np.cos(angle_y)],
    ])
    rx = np.array([
        [1, 0, 0],
        [0, np.cos(angle_x), -np.sin(angle_x)],
        [0, np.sin(angle_x), np.cos(angle_x)],
    ])
    r = rx @ ry
    projected = points_xyz @ r.T
    
    # Scale to compensate for foreshortening
    # For 45° isometric view, scale X by ~1.22 to restore proportions
    xy = projected[:, :2].copy()
    xy[:, 0] *= 1.22
    return xy



def _normalize_vec(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return v
    return v / n


def _alignment_joints(skeleton_format: str) -> tuple[int, tuple[int, int], tuple[int, int]]:
    """
    Returns:
      - head joint index
      - (left_foot, right_foot) joint indices
      - (left_shoulder, right_shoulder) joint indices
    """
    if skeleton_format == "ntu25":
        # Kinect v2 / NTU:
        # head=3, feet=(15,19), shoulders=(4,8)
        return 3, (15, 19), (4, 8)
    if skeleton_format == "h36m17":
        # Human3.6M (Martinez ordering):
        # head=10, ankles=(6,3), shoulders=(11,14)
        return 10, (6, 3), (11, 14)
    raise ValueError(f"Unknown skeleton_format: {skeleton_format}")


def _compute_alignment_transform(
    skeleton_xyz: np.ndarray,
    *,
    skeleton_format: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a rigid transform (rotation + origin) so that:
      - feet->head is aligned with +Y axis
      - left->right shoulders is aligned with +X axis (at initial pose)

    Returns:
      - origin (3,) to subtract before rotation (feet midpoint)
      - R (3,3) rotation matrix where columns are the new basis (x,y,z)
    """
    head_idx, (lf_idx, rf_idx), (ls_idx, rs_idx) = _alignment_joints(skeleton_format)
    pts = np.asarray(skeleton_xyz, dtype=np.float32)
    if pts.ndim != 3 or pts.shape[-1] != 3:
        raise ValueError(f"Expected skeleton_xyz shape (T, J, 3); got {pts.shape}")

    feet_mid = 0.5 * (pts[:, lf_idx, :] + pts[:, rf_idx, :])  # (T,3)
    head = pts[:, head_idx, :]  # (T,3)
    left_sh = pts[:, ls_idx, :]  # (T,3)
    right_sh = pts[:, rs_idx, :]  # (T,3)

    # Use average direction across the entire sequence (more stable than frame-0),
    # but normalize per-frame before averaging to avoid cancellations when the
    # subject turns in place (shoulder vector rotates around Y).
    origin = feet_mid.mean(axis=0)

    y_frames = head - feet_mid  # (T,3)
    y_frames = np.stack([_normalize_vec(v) for v in y_frames], axis=0)
    y_axis = _normalize_vec(y_frames.mean(axis=0))  # feet -> head
    if float(np.linalg.norm(y_axis)) == 0.0:
        y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    # X is based on per-frame shoulder vectors, sign-aligned, then averaged.
    x_frames = right_sh - left_sh  # (T,3)
    # Remove any component along Y (Gram-Schmidt per-frame)
    x_frames = x_frames - (x_frames @ y_axis)[:, None] * y_axis[None, :]
    x_frames = np.stack([_normalize_vec(v) for v in x_frames], axis=0)

    # Choose a reference direction (first non-degenerate frame) and flip
    # subsequent frames to match it, preventing mean cancellation.
    ref = None
    for v in x_frames:
        if float(np.linalg.norm(v)) > 0.0:
            ref = v
            break
    if ref is None:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    signs = np.sign(x_frames @ ref)
    signs[signs == 0] = 1
    x_frames = x_frames * signs[:, None]

    x_axis = _normalize_vec(x_frames.mean(axis=0))
    if float(np.linalg.norm(x_axis)) == 0.0:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    z_axis = np.cross(x_axis, y_axis)
    z_axis = _normalize_vec(z_axis)
    if float(np.linalg.norm(z_axis)) == 0.0:
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # Re-orthogonalize x to guarantee right-handed basis
    x_axis = _normalize_vec(np.cross(y_axis, z_axis))

    r = np.stack([x_axis, y_axis, z_axis], axis=1)  # columns are basis vectors
    return origin, r


def _project_skeleton_aligned(
    points_xyz: np.ndarray,
    *,
    origin: np.ndarray,
    r: np.ndarray,
    skeleton_format: str,  # Add this parameter
) -> np.ndarray:
    """
    Project 3D points to 2D after aligning to a canonical pose.
    """
    pts = np.asarray(points_xyz, dtype=np.float32) - origin[None, :]
    aligned = pts @ r  # (J,3)
    # Project isometrically (same style as `skeleton_render_frames.py`).
    return _apply_isometric_projection(aligned)


def _load_skeleton_npz(npz_path: Path) -> dict[str, np.ndarray]:
    # NPZ is expected to be keyed by clip filename, as in `skeleton_render_frames.py`
    return {str(k): v for k, v in dict(np.load(npz_path)).items()}


def _normalize_skeleton_array(arr: np.ndarray, skeleton_format: str) -> np.ndarray:
    """
    Normalize skeleton array to shape (T, J, 3) depending on skeleton_format.
    - ntu25: (T, 75) or (T, 25, 3)
    - h36m17: (T, 51) or (T, 17, 3)
    """
    a = np.asarray(arr)
    if skeleton_format == "ntu25":
        if a.ndim == 2 and a.shape[1] == 25 * 3:
            a = a.reshape(a.shape[0], 25, 3)
        elif a.ndim == 3 and a.shape[1:] == (25, 3):
            pass
        else:
            raise ValueError(f"Unexpected ntu25 skeleton shape {a.shape}; expected (T, 75) or (T, 25, 3).")
    elif skeleton_format == "h36m17":
        if a.ndim == 2 and a.shape[1] == 17 * 3:
            a = a.reshape(a.shape[0], 17, 3)
        elif a.ndim == 3 and a.shape[1:] == (17, 3):
            pass
        else:
            raise ValueError(f"Unexpected h36m17 skeleton shape {a.shape}; expected (T, 51) or (T, 17, 3).")
    else:
        raise ValueError(f"Unknown skeleton_format: {skeleton_format}")

    # Match the sign convention used in `skeleton_render_frames.py` (flip Y).
    a = a.copy()
    a[:, :, 1] *= -1
    return a


def _find_skeleton_key(skel_dict: dict[str, np.ndarray], clip_id: str) -> Optional[str]:
    if clip_id in skel_dict:
        return clip_id
    if f"{clip_id}.mp4" in skel_dict:
        return f"{clip_id}.mp4"
    if f"{clip_id}.MP4" in skel_dict:
        return f"{clip_id}.MP4"
    return None


def _plot_sliding_window_row(
    predictions: np.ndarray,
    *,
    out_path: Path,
    window_size: int,
    stride: int,
    fps: int,
    threshold: float,
    seq_len_frames: Optional[int],
    skeleton_xyz: Optional[np.ndarray],
    skeleton_connections: Optional[list[tuple[int, int]]],
    skeleton_format: Optional[str],
) -> None:
    # Lazy import so analysis-only usage doesn't require matplotlib installed.
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    # Window "center" positions in frames -> seconds
    window_pos_frames = (np.arange(predictions.shape[0], dtype=np.float32) * float(stride)) + (float(window_size) / 2.0)
    window_pos_sec = window_pos_frames / float(fps)

    if seq_len_frames is None:
        # Best-effort: last window start + window size
        if predictions.size == 0:
            seq_len_frames = window_size
        else:
            seq_len_frames = int((predictions.shape[0] - 1) * stride + window_size)
    seq_len_sec = float(seq_len_frames) / float(fps)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(0.0, max(seq_len_sec, 1e-6))
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("P(unstable | window)")

    mask = predictions > threshold
    ax.plot(window_pos_sec, predictions, color="gray", linewidth=1)
    ax.axhline(
        threshold,
        alpha=0.6,
        color=(0, 0.5, 0),
        linestyle="--",
        linewidth=1.5,
        label="unstable threshold",
    )

    bar_width = float(window_size) / float(fps)
    # Draw window rectangles starting from the decision threshold:
    # - stable windows go from threshold downward (negative height)
    # - unstable windows go from threshold upward (positive height)
    stable_heights = predictions[~mask] - threshold
    unstable_heights = predictions[mask] - threshold
    ax.bar(
        window_pos_sec[~mask],
        stable_heights,
        width=bar_width,
        bottom=threshold,
        color="blue",
        alpha=0.15,
        align="center",
    )
    ax.scatter(window_pos_sec[~mask], predictions[~mask], color="blue", label="stable (0)", s=10)
    ax.bar(
        window_pos_sec[mask],
        unstable_heights,
        width=bar_width,
        bottom=threshold,
        color="red",
        alpha=0.15,
        align="center",
    )
    ax.scatter(window_pos_sec[mask], predictions[mask], color="red", label="unstable (1)", s=10)

    ax.legend(frameon=False)
    ax.grid()
    fig.tight_layout()

    if skeleton_xyz is not None and skeleton_connections is not None and predictions.size > 0:
        if skeleton_format is None:
            raise ValueError("skeleton_format is required when plotting skeleton overlays.")

        # Compute a canonical 3D alignment (body-aligned XYZ), then project isometrically.
        origin, r = _compute_alignment_transform(skeleton_xyz, skeleton_format=skeleton_format)

        # Draw tiny skeletons near each window marker.
        # IMPORTANT: skeleton size must remain uniform across the entire clip; only placement changes.
        fig.canvas.draw()
        ax_bbox = ax.get_window_extent()
        x_range = float(ax.get_xlim()[1] - ax.get_xlim()[0])
        y_range = float(ax.get_ylim()[1] - ax.get_ylim()[0])
        x_units_per_px = x_range / max(float(ax_bbox.width), 1.0)
        y_units_per_px = y_range / max(float(ax_bbox.height), 1.0)

        # Fixed on-screen skeleton size (in pixels).
        target_width_px = 120.0
        target_height_px = 150.0
        x_span = x_units_per_px * target_width_px   # seconds
        y_span = y_units_per_px * target_height_px  # probability units

        clear_px = 15.0  # keep head/feet from touching the marker exactly
        clear_y = y_units_per_px * clear_px

        head_idx, (lf_idx, rf_idx), _ = _alignment_joints(skeleton_format)

        t_max = int(skeleton_xyz.shape[0]) - 1
        # Precompute a single normalization factor across all plotted midpoints so
        # skeleton size stays constant across the clip.
        frame_idxs = [max(0, min(t_max, int(round(float(f))))) for f in window_pos_frames.tolist()]
        uniq_frame_idxs = sorted(set(frame_idxs))
        frame_cache: dict[int, tuple[np.ndarray, float]] = {}
        denom = 0.0
        for fi in uniq_frame_idxs:
            joints_xy = _project_skeleton_aligned(
                skeleton_xyz[fi],
                origin=origin,
                r=r,
                skeleton_format=skeleton_format,
            ).astype(np.float32)
            feet_mid_xy = 0.5 * (joints_xy[lf_idx] + joints_xy[rf_idx])
            xy_local = joints_xy - feet_mid_xy  # feet midpoint at (0,0)
            frame_cache[fi] = (xy_local, float(xy_local[head_idx, 1]))
            denom = max(denom, float(np.max(np.abs(xy_local))))
        if denom == 0.0:
            denom = 1.0

        for k, frame_idx in enumerate(frame_idxs):
            xy_local, head_y_local = frame_cache[frame_idx]
            xy = xy_local / float(denom)  # unitless, consistent across all windows
            head_y_norm = float(head_y_local) / float(denom)

            x0 = float(window_pos_sec[k])
            y0 = float(predictions[k])

            # Placement rule:
            # - If marker > 0.5: skeleton below marker, with head right below marker.
            # - Else: skeleton above marker, with feet right above marker.
            if y0 > threshold:
                y_anchor = (y0 - clear_y) - (head_y_norm * y_span)
            else:
                y_anchor = (y0 + clear_y)  # feet are at y=0 in local coords

            pts = np.empty_like(xy)
            pts[:, 0] = x0 + xy[:, 0] * x_span
            pts[:, 1] = y_anchor + xy[:, 1] * y_span

            segments = [[pts[i], pts[j]] for i, j in skeleton_connections]
            lc = LineCollection(segments, colors=[(0, 0, 0, 1.0)], linewidths=1.0, zorder=3)
            ax.add_collection(lc)
            ax.scatter(pts[:, 0], pts[:, 1], c=[(0, 0, 0, 1.00)], s=4, zorder=3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def _maybe_add_subject_column(df: pd.DataFrame, mapping_csv: Optional[Path]) -> pd.DataFrame:
    # Preferred: subject already in the CSV
    if "subject" in df.columns:
        return df.assign(subject=df["subject"].astype(int))

    # Next: mapping file for the public sarcopenia dataset (index -> subject id)
    if mapping_csv is not None:
        mapping_df = pd.read_csv(mapping_csv)
        if not {"index", "subject"}.issubset(mapping_df.columns):
            raise SystemExit("Mapping CSV must have 'index' and 'subject' columns.")
        index_to_subject = dict(zip(mapping_df["index"].astype(int), mapping_df["subject"].astype(int)))
        if "index" not in df.columns:
            raise SystemExit("Input CSV is missing 'index' column; can't apply mapping CSV.")
        subjects = df["index"].astype(int).map(index_to_subject)
        if subjects.isna().any():
            missing = df.loc[subjects.isna(), "index"].astype(int).unique().tolist()
            raise SystemExit(f"Missing subject mapping for index values: {missing[:10]}{'...' if len(missing) > 10 else ''}")
        return df.assign(subject=subjects.astype(int))

    # Fallback: treat the row index as the subject identifier
    if "index" in df.columns:
        return df.assign(subject=df["index"].astype(int))
    return df.assign(subject=pd.RangeIndex(len(df)).astype(int))


def compute_subject_instability(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    if "predictions" not in df.columns:
        raise SystemExit("Input CSV must contain a 'predictions' column.")

    preds = df["predictions"].map(_ensure_predictions_array)
    unstable_count = preds.map(lambda a: int((a > threshold).sum()))
    stable_count = preds.map(lambda a: int((a <= threshold).sum()))

    denom = (stable_count + unstable_count).to_numpy()
    instability_rate = np.where(denom > 0, unstable_count / (stable_count + unstable_count), np.nan)

    out = df.copy()
    out["instability_rate"] = instability_rate
    out["risk_group"] = np.where(out["instability_rate"].to_numpy() >= 0.5, "mostly unstable", "mostly stable")
    out.loc[out["instability_rate"].isna(), "risk_group"] = "unknown"

    # Ensure a clip identifier exists for printing
    if "clip" not in out.columns:
        if "index" in out.columns:
            out["clip"] = out["index"].astype(str)
        else:
            out["clip"] = out.index.astype(str)

    # Print one row per clip (i.e., per input sample)
    cols = ["clip"]
    if "subject" in out.columns:
        cols.append("subject")
    cols += ["instability_rate", "risk_group"]
    return out[cols]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-subject instability rate (R) and sarcopenia risk group from a single "
            "sliding-window CSV produced by `skateformer/inference.py`."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to a sliding-window CSV, or a work_dir containing exactly one such CSV.",
    )
    parser.add_argument(
        "--mapping-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV mapping `index` -> `subject` (e.g. `csvs/sliding_window_subject_idx_to_subject_id.csv`). "
            "Used if the input CSV does not already contain a `subject` column."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output CSV path. If omitted, prints to stdout.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold applied to per-window probability to count a window as unstable. Default: 0.5.",
    )
    parser.add_argument(
        "--plots-out",
        type=Path,
        default=None,
        help=(
            "Optional directory to write per-clip sliding-window plots (PDF). "
            "If omitted, sliding-window plots are not generated."
        ),
    )
    parser.add_argument(
        "--skeleton-npz",
        type=Path,
        default=None,
        help=(
            "Optional NPZ of skeleton sequences (same format as `skeleton_render_frames.py`, keyed by clip filename). "
            "If provided along with --plots-out, tiny skeletons are overlaid at each window midpoint."
        ),
    )
    parser.add_argument(
        "--skeleton-format",
        choices=["ntu25", "h36m17"],
        default=None,
        help="Skeleton layout stored in --skeleton-npz. Required if --skeleton-npz is set.",
    )
    args = parser.parse_args()

    if args.skeleton_npz is not None and args.skeleton_format is None:
        raise SystemExit("--skeleton-format is required when --skeleton-npz is set.")

    csv_path = _resolve_sliding_window_csv(args.input)
    df = pd.read_csv(csv_path)
    df = _maybe_add_subject_column(df, args.mapping_csv)
    df["predictions"] = df["predictions"].map(_ensure_predictions_array)
    out_df = compute_subject_instability(df, threshold=args.threshold)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.out, index=False)
        print(f"Wrote: {args.out}")
    else:
        # Compact deterministic printing
        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 200):
            print(out_df.to_string(index=False))

    if args.plots_out is not None:
        from tqdm import tqdm

        skel_dict: Optional[dict[str, np.ndarray]] = None
        if args.skeleton_npz is not None:
            skel_dict = _load_skeleton_npz(args.skeleton_npz)

        window_size, stride = _parse_window_and_stride_from_name(csv_path)
        if window_size is None or stride is None:
            raise SystemExit(
                "Could not infer window_size/stride from filename. Expected something like "
                "`sliding_window_windowsize64_stride8.csv`."
            )

        # Use an "effective stride" equal to the window size (non-overlapping windows).
        # If the file was created with a smaller stride (e.g. 8), subsample by taking
        # one window and skipping the others (e.g. take every 8th window when window_size=64).
        effective_stride = int(window_size)
        if effective_stride % int(stride) != 0:
            raise SystemExit(
                f"Effective stride {effective_stride} must be a multiple of file stride {stride} "
                f"to subsample windows."
            )
        stride_step = effective_stride // int(stride)

        # Use a stable clip identifier for filenames (prefer clip, then index, then row number).
        clip_series = df["clip"] if "clip" in df.columns else (df["index"].astype(str) if "index" in df.columns else df.index.astype(str))
        for i in tqdm(range(len(df)), desc="Plotting"):
            clip_id = str(clip_series.iloc[i])
            out_name = f"{_safe_stem(clip_id)}_stability_plot.png"
            out_path = args.plots_out / out_name

            row = df.iloc[i]
            seq_len_frames: Optional[int] = None
            if "seq_len" in df.columns:
                try:
                    seq_len_frames = int(row["seq_len"])
                except Exception:
                    seq_len_frames = None

            preds: np.ndarray = row["predictions"]
            if stride_step > 1:
                preds = preds[::stride_step]

            skeleton_xyz: Optional[np.ndarray] = None
            skeleton_connections: Optional[list[tuple[int, int]]] = None
            if skel_dict is not None:
                key = _find_skeleton_key(skel_dict, clip_id)
                if key is not None:
                    try:
                        skeleton_xyz = _normalize_skeleton_array(skel_dict[key], args.skeleton_format)
                        skeleton_connections = (
                            _BONE_CONNECTIONS_NTU25 if args.skeleton_format == "ntu25" else _BONE_CONNECTIONS_H36M17
                        )
                    except Exception:
                        skeleton_xyz = None
                        skeleton_connections = None

            _plot_sliding_window_row(
                preds,
                out_path=out_path,
                window_size=window_size,
                stride=effective_stride,
                fps=30,
                threshold=float(args.threshold),
                seq_len_frames=seq_len_frames,
                skeleton_xyz=skeleton_xyz,
                skeleton_connections=skeleton_connections,
                skeleton_format=args.skeleton_format,
            )


if __name__ == "__main__":
    main()
