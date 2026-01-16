from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Iterable, List, Sequence

import cv2
import einops
# import tkinter
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Path3DCollection
from seq_transformation import seq_translation

SKELETON_CONNECTIONS_KINECT = np.array(
    [
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
    ],
    dtype=int,
)

# Human3.6M 17-joint layout (Martinez et al. ordering: 0–16).
# 0:pelvis, 1:r-hip, 2:r-knee, 3:r-ankle, 4:l-hip, 5:l-knee, 6:l-ankle,
# 7:spine, 8:thorax, 9:neck/nose, 10:head, 11:l-shoulder, 12:l-elbow,
# 13:l-wrist, 14:r-shoulder, 15:r-elbow, 16:r-wrist
SKELETON_CONNECTIONS_H36M = np.array(
    [
        [0, 1],
        [1, 2],
        [2, 3],
        [0, 4],
        [4, 5],
        [5, 6],
        [0, 7],
        [7, 8],
        [8, 9],
        [9, 10],
        [8, 11],
        [11, 12],
        [12, 13],
        [8, 14],
        [14, 15],
        [15, 16],
    ],
    dtype=int,
)

SKELETON_CONNECTION_MAP = {
    "kinect_v2": SKELETON_CONNECTIONS_KINECT,
    "h36m": SKELETON_CONNECTIONS_H36M,
}
COLOR_MAP = plt.cm.get_cmap("tab10")


def _ensure_skeleton_list(
    skeletons: Sequence[np.ndarray] | np.ndarray,
) -> List[np.ndarray]:
    if isinstance(skeletons, np.ndarray):
        return [skeletons]
    if isinstance(skeletons, Iterable):
        return [np.asarray(sk) for sk in skeletons]
    raise TypeError("skeletons must be an ndarray or an iterable of ndarrays.")


def _normalize_skeleton(skeleton: np.ndarray) -> np.ndarray:
    if skeleton.ndim == 3 and skeleton.shape[-1] == 3:
        skeleton = einops.rearrange(skeleton, "frame joint coord -> frame (joint coord)")
    # skeleton = seq_translation(skeleton[None])[0]
    skeleton = einops.rearrange(skeleton, "frame (joint coord) -> frame joint coord", coord=3)

    # metrabs:
    # skeleton -= skeleton[0,0,:]
    # skeleton = skeleton[:,:,[0,2,1]]
    # skeleton[:,:,2] *= -1
    return skeleton


def visualize_skeleton(
    skeletons: Sequence[np.ndarray] | np.ndarray,
    clip_names: Sequence[str] | None = None,
    video_root: Path | None = None,
    repeat: bool = True,
    figure: plt.Figure | None = None,
    block: bool = True,
    skeleton_type: str = "kinect_v2",
) -> None:
    if skeleton_type not in SKELETON_CONNECTION_MAP:
        raise ValueError(f"Unknown skeleton_type '{skeleton_type}'. Expected one of {list(SKELETON_CONNECTION_MAP)}.")
    connections = SKELETON_CONNECTION_MAP[skeleton_type]

    skels = _ensure_skeleton_list(skeletons)
    if not skels:
        raise ValueError("No skeletons provided.")

    skels = [_normalize_skeleton(skel) for skel in skels]
    clip_names = list(clip_names) if clip_names is not None else [f"clip_{i}" for i in range(len(skels))]
    if len(clip_names) != len(skels):
        raise ValueError("clip_names must match the number of skeletons.")

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
    skel_lines = [skel[:, connections, :] for skel in skels]

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

    S = 1
    ax.set_xlim(-S, S)
    ax.set_ylim(-S, S)
    ax.set_zlim(-S+1, S+1)
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

    for idx, skel in enumerate(skels):
        color = COLOR_MAP(idx % COLOR_MAP.N)
        pts = ax.scatter(*skel[0].T, color=color, s=10)
        line = Line3DCollection(skel_lines[idx][0], colors=[color], linewidths=2)
        ax.add_collection3d(line)
        points.append(pts)
        lines.append(line)
        legend_handles.append(Line2D([0], [0], color=color, lw=2, label=clip_names[idx]))

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
    parser = argparse.ArgumentParser(description="Visualize skeleton sequences alongside source videos.")
    parser = argparse.ArgumentParser(description="Visualize skeleton sequences alongside source videos.")
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
    parser.add_argument(
        "--skeleton-type",
        choices=list(SKELETON_CONNECTION_MAP.keys()),
        default="kinect_v2",
        help="Skeleton layout to use when drawing (default: kinect_v2).",
    )
    args = parser.parse_args()

    skel_dict = dict(np.load(args.npz_path))
    if not skel_dict:
        raise ValueError(f"No skeletons found in {args.npz_path}")

    fig = plt.figure()
    for clip in skel_dict:
        visualize_skeleton(
            [skel_dict[clip]],
            clip_names=[clip],
            video_root=args.video_root,
            repeat=False,
            figure=fig,
            skeleton_type=args.skeleton_type,
        )


if __name__ == "__main__":
    main()
