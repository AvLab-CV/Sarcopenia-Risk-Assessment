from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Iterable, List, Sequence, Dict, Set

import cv2
import einops
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - needed for 3D projection
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Path3DCollection

from seq_transformation import seq_translation
from zeni import SkeletonGaitAnalysis


# Skeleton layout (NTU-like, matching visualize_skeleton.py)
SKELETON_CONNECTIONS = [
    # Torso
    [0, 1],
    [1, 20],
    [20, 2],
    [2, 3],
    # Left Arm
    [2, 4],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 21],
    [7, 22],
    # Right Arm
    [2, 8],
    [8, 9],
    [9, 10],
    [10, 11],
    [11, 23],
    [11, 24],
    # Left Leg
    [0, 12],
    [12, 13],
    [13, 14],
    [14, 15],
    # Right Leg
    [0, 16],
    [16, 17],
    [17, 18],
    [18, 19],
]
SKELETON_CONNECTIONS = np.array(SKELETON_CONNECTIONS)
COLOR_MAP = plt.cm.get_cmap("tab10")

# Joint indices for gait analysis (NTU convention)
SPINE_BASE = 0  # pelvis / sacrum proxy
LEFT_ANKLE = 14
LEFT_FOOT = 15
RIGHT_ANKLE = 18
RIGHT_FOOT = 19


def _ensure_skeleton_list(
    skeletons: Sequence[np.ndarray] | np.ndarray,
) -> List[np.ndarray]:
    if isinstance(skeletons, np.ndarray):
        return [skeletons]
    if isinstance(skeletons, Iterable):
        return [np.asarray(sk) for sk in skeletons]
    raise TypeError("skeletons must be an ndarray or an iterable of ndarrays.")


def _normalize_skeleton(skeleton: np.ndarray) -> np.ndarray:
    """Match the normalization used in visualize_skeleton.py.

    - Accepts (T, J, 3) or (T, J*3)
    - Translates sequence so root joint is centered using seq_translation.
    """
    if skeleton.ndim == 3 and skeleton.shape[-1] == 3:
        skeleton = einops.rearrange(skeleton, "frame joint coord -> frame (joint coord)")
    skeleton = seq_translation(skeleton[None])[0]
    return einops.rearrange(skeleton, "frame (joint coord) -> frame joint coord", coord=3)


def visualize_skeleton_with_gait_events(
    skeletons: Sequence[np.ndarray] | np.ndarray,
    clip_names: Sequence[str] | None = None,
    video_root: Path | None = None,
    repeat: bool = True,
    figure: plt.Figure | None = None,
    block: bool = True,
) -> None:
    """Visualize skeleton(s) and overlay gait events estimated via Zeni et al.

    Arguments mirror visualize_skeleton.visualize_skeleton.
    - HS/TO events are shown on feet:
        * Left HS:  red marker on left ankle
        * Left TO:  blue marker on left foot
        * Right HS: magenta marker on right ankle
        * Right TO: cyan marker on right foot
    """

    skels = _ensure_skeleton_list(skeletons)
    if not skels:
        raise ValueError("No skeletons provided.")

    # Normalize skeletons as in the original visualizer
    skels = [_normalize_skeleton(skel) for skel in skels]

    clip_names = list(clip_names) if clip_names is not None else [f"clip_{i}" for i in range(len(skels))]
    if len(clip_names) != len(skels):
        raise ValueError("clip_names must match the number of skeletons.")

    # --- Video lookup (same behaviour as visualize_skeleton.py) ---
    show_video = video_root is not None
    video_lookup: dict[str, Path] = {}
    if show_video:
        if video_root.exists():
            video_lookup = {p.name: p for p in video_root.rglob("*.mp4")}
        else:
            print(f"Video root {video_root} not found; showing skeletons only.")
            show_video = False

    video_paths = [video_lookup.get(name) for name in clip_names]
    primary_video_idx = next((idx for idx, path in enumerate(video_paths) if path and path.exists()), None)

    max_frames = max(skel.shape[0] for skel in skels)
    skel_lines = [skel[:, SKELETON_CONNECTIONS, :] for skel in skels]

    fig = figure if figure is not None else plt.figure()
    saved_view = None
    if fig.axes:
        for existing_ax in fig.axes:
            if getattr(existing_ax, "name", "") == "3d":
                saved_view = {
                    "elev": getattr(existing_ax, "elev", None),
                    "azim": getattr(existing_ax, "azim", None),
                    "roll": getattr(existing_ax, "roll", None),
                }
                break
    fig.clf()
    if show_video and primary_video_idx is not None:
        ax: Axes3D = fig.add_subplot(1, 2, 1, projection="3d")
        ax2 = fig.add_subplot(1, 2, 2)
        image = ax2.imshow(np.zeros((10, 10, 3)))
    else:
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax2 = image = None

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(0, 2)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if saved_view and saved_view["elev"] is not None and saved_view["azim"] is not None:
        view_kwargs = {"elev": saved_view["elev"], "azim": saved_view["azim"]}
        if saved_view["roll"] is not None:
            view_kwargs["roll"] = saved_view["roll"]
        try:
            ax.view_init(**view_kwargs)
        except TypeError:
            ax.view_init(elev=view_kwargs["elev"], azim=view_kwargs["azim"])

    points: List[Path3DCollection] = []
    lines: List[Line3DCollection] = []
    legend_handles: List[Line2D] = []

    # --- Gait analysis (Zeni) ---
    gait = SkeletonGaitAnalysis(fps=30)
    gait_events: List[Dict[str, Set[int]]] = []

    for idx, skel in enumerate(skels):
        # Estimate events for left and right legs
        sacrum_pos = skel[:, SPINE_BASE, :]

        l_heel = skel[:, LEFT_ANKLE, :]
        l_toe = skel[:, LEFT_FOOT, :]
        r_heel = skel[:, RIGHT_ANKLE, :]
        r_toe = skel[:, RIGHT_FOOT, :]

        l_hs, l_to = gait.detect_events_zeni(l_heel, l_toe, sacrum_pos)
        r_hs, r_to = gait.detect_events_zeni(r_heel, r_toe, sacrum_pos)

        events_for_clip: Dict[str, Set[int]] = {
            "l_hs": set(l_hs.tolist()),
            "l_to": set(l_to.tolist()),
            "r_hs": set(r_hs.tolist()),
            "r_to": set(r_to.tolist()),
        }
        gait_events.append(events_for_clip)

        # Quick textual summary in frames
        print(f"Clip '{clip_names[idx]}':")
        print(f"  Left HS frames:  {sorted(events_for_clip['l_hs'])}")
        print(f"  Left TO frames:  {sorted(events_for_clip['l_to'])}")
        print(f"  Right HS frames: {sorted(events_for_clip['r_hs'])}")
        print(f"  Right TO frames: {sorted(events_for_clip['r_to'])}")

    # --- Base skeleton artists ---
    for idx, skel in enumerate(skels):
        color = COLOR_MAP(idx % COLOR_MAP.N)
        pts = ax.scatter(*skel[0].T, color=color, s=10)
        line = Line3DCollection(skel_lines[idx][0], colors=[color], linewidths=2)
        ax.add_collection3d(line)
        points.append(pts)
        lines.append(line)
        legend_handles.append(Line2D([0], [0], color=color, lw=2, label=clip_names[idx]))

    # --- Event markers (always created; shown only on event frames) ---
    event_markers: List[Dict[str, Path3DCollection]] = []
    for _ in skels:
        # Empty scatters that we'll move frame-by-frame
        markers_for_clip = {
            "l_hs": ax.scatter([], [], [], color="red", s=40, marker="o"),
            "l_to": ax.scatter([], [], [], color="blue", s=40, marker="o"),
            "r_hs": ax.scatter([], [], [], color="magenta", s=40, marker="o"),
            "r_to": ax.scatter([], [], [], color="cyan", s=40, marker="o"),
        }
        event_markers.append(markers_for_clip)

    # Track printed events so we don't spam during looping animations
    printed_events: List[Set[tuple[str, int]]] = [set() for _ in skels]

    if len(legend_handles) > 1:
        ax.legend(handles=legend_handles, loc="upper right")

    cap = None
    if show_video and primary_video_idx is not None:
        video_path = video_paths[primary_video_idx]
        cap = cv2.VideoCapture(str(video_path))
        fig.suptitle(f"{', '.join(clip_names)} (video: {clip_names[primary_video_idx]})", fontsize=14)
    else:
        fig.suptitle(", ".join(clip_names), fontsize=14)

    def _next_video_frame():
        if cap is None or image is None:
            return
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        if ret:
            frame = frame[:, :, [2, 1, 0]]
            image.set_data(frame)

    def animate(frame_idx: int):
        idx = frame_idx % max_frames
        for skel_idx, skel in enumerate(skels):
            local_idx = min(idx, skel.shape[0] - 1)
            lines[skel_idx].set_segments(skel_lines[skel_idx][local_idx])
            points[skel_idx]._offsets3d = (
                skel[local_idx, :, 0],
                skel[local_idx, :, 1],
                skel[local_idx, :, 2],
            )

            # Update gait event markers for this skeleton
            events = gait_events[skel_idx]
            markers = event_markers[skel_idx]
            seen = printed_events[skel_idx]
            clip_name = clip_names[skel_idx]

            # Helper: show/hide marker at given joint if current frame is an event
            def _update_marker(marker_key: str, joint_idx: int, label: str):
                marker = markers[marker_key]
                if local_idx in events[marker_key]:
                    pos = skel[local_idx, joint_idx]
                    marker._offsets3d = ([pos[0]], [pos[1]], [pos[2]])

                    # Print once per event occurrence
                    key = (marker_key, local_idx)
                    if key not in seen:
                        step_length = None
                        if marker_key in {"l_hs", "r_hs"}:
                            l_pos = skel[local_idx, LEFT_ANKLE, :]
                            r_pos = skel[local_idx, RIGHT_ANKLE, :]
                            step_length = float(np.linalg.norm(l_pos - r_pos))

                        if step_length is not None:
                            print(
                                f"[{clip_name}] frame {local_idx}: {label} | step_length={step_length:.4f}"
                            )
                        else:
                            print(f"[{clip_name}] frame {local_idx}: {label}")
                        seen.add(key)
                else:
                    # Hide by moving to empty data
                    marker._offsets3d = ([], [], [])

            _update_marker("l_hs", LEFT_ANKLE, "Left HS")
            _update_marker("l_to", LEFT_FOOT, "Left TO")
            _update_marker("r_hs", RIGHT_ANKLE, "Right HS")
            _update_marker("r_to", RIGHT_FOOT, "Right TO")

        _next_video_frame()

    interval_ms = 30
    if repeat:
        ani: FuncAnimation | None = FuncAnimation(
            fig, animate, frames=itertools.count(), interval=interval_ms, blit=False, repeat=True
        )
    else:
        ani = None

    def _cleanup(_):
        if cap is not None:
            cap.release()

    fig.canvas.mpl_connect("close_event", _cleanup)
    if repeat and ani is not None:
        ani.resume()
        plt.show(block=block)
        return

    # For non-repeating playback, drive frames manually to avoid timer callback issues.
    plt.show(block=False)
    for frame_idx in range(max_frames):
        animate(frame_idx)
        fig.canvas.draw_idle()
        plt.pause(interval_ms / 1000)
    _cleanup(None)
    if block:
        plt.pause(0.2)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize skeleton sequences with gait events (Zeni) alongside source videos."
    )
    parser.add_argument(
        "npz_path",
        type=Path,
        help="Path to the NPZ file containing skeleton arrays (keyed by clip filename).",
    )
    parser.add_argument(
        "video_root",
        type=Path,
        nargs="?",
        help="Optional root directory containing source .mp4 files.",
    )
    args = parser.parse_args()

    skel_dict = dict(np.load(args.npz_path))
    if not skel_dict:
        raise ValueError(f"No skeletons found in {args.npz_path}")

    fig = plt.figure()
    for clip in itertools.cycle(skel_dict):
        visualize_skeleton_with_gait_events(
            [skel_dict[clip]],
            clip_names=[clip],
            video_root=args.video_root,
            repeat=False,
            figure=fig,
        )


if __name__ == "__main__":
    main()



