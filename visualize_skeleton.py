import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Path3DCollection
import numpy as np
from matplotlib.animation import FuncAnimation
import einops
import cv2
from pathlib import Path
from seq_transformation import seq_translation

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


def main():
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
    args = parser.parse_args()

    skel_dict = dict(np.load(args.npz_path))
    if not skel_dict:
        raise ValueError(f"No skeletons found in {args.npz_path}")

    clip_names = sorted(skel_dict.keys())
    skels = [skel_dict[name] for name in clip_names]

    show_video = args.video_root is not None
    if show_video and args.video_root.exists():
        video_lookup = {p.name: p for p in args.video_root.rglob("*.mp4")}
    elif show_video:
        print(f"Video root {args.video_root} not found; showing skeleton only.")
        video_lookup = {}
    else:
        video_lookup = {}
    videos = [video_lookup.get(name) if show_video else None for name in clip_names]

    anim_idx = 0
    base_idx = 0

    for i in range(len(skels)):
        # Seq translation
        skels[i] = seq_translation(skels[i][None])[0]
        
        # Unmerge joint and coord dims
        skels[i] = einops.rearrange(skels[i], "frame (joint coord) -> frame joint coord", coord=3)
        
        # Swap Y/Z to match visualization expectations
        # skels[i] = skels[i][:, :, [0, 2, 1]]
        # skels[i][:, :, 2] *= -1.0
        # skels[i][:, :, 2] += 1

    skels_lines = [skel[:, SKELETON_CONNECTIONS, :] for skel in skels]

    def init_capture(index: int):
        path = videos[index]
        if path is None or not path.exists():
            print(f"Video file not found for {clip_names[index]}; showing skeleton only.")
            return None, None, None, None, None
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        return cap, fps, width, height, frame_count
    
    cap = fps = width = height = frame_count = None
    if show_video:
        cap, fps, width, height, frame_count = init_capture(anim_idx)

    clip_msg = f"Clip: {clip_names[anim_idx]} skel_frames={skels[anim_idx].shape[0]}"
    if show_video and fps is not None:
        clip_msg += f" FPS={fps} frames={frame_count} width={width} height={height}"
    print(clip_msg)

    fig = plt.figure()
    if show_video:
        ax: Axes3D = fig.add_subplot(1, 2, 1, projection="3d")
    else:
        ax: Axes3D = fig.add_subplot(1, 1, 1, projection="3d")
    points: Path3DCollection = ax.scatter(*skels[anim_idx][0].T)  # Points

    if show_video:
        ax2: Axes3D = fig.add_subplot(1, 2, 2)
        image = ax2.imshow(np.zeros((10, 10, 3)))
    else:
        ax2 = image = None

    lines = Line3DCollection(skels_lines[anim_idx][0])
    ax.add_collection3d(lines)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(0, 2)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    def animate(i):
        nonlocal anim_idx, base_idx, cap, fps, frame_count

        frame = i - base_idx
        if frame >= skels[anim_idx].shape[0]:
            base_idx = i
            anim_idx += 1
            if anim_idx >= len(skels):
                anim_idx = 0
            frame = 0
            print(f"Next animation: {anim_idx}")
            if cap is not None:
                cap.release()
            if show_video:
                cap, fps, width, height, frame_count = init_capture(anim_idx)
            clip_summary = f"Clip: {clip_names[anim_idx]} skel_frames={skels[anim_idx].shape[0]}"
            if show_video and fps is not None:
                clip_summary += f" FPS={fps} frames={frame_count} width={width} height={height}"
            print(clip_summary)

        skel = skels[anim_idx]
        skel_lines = skels_lines[anim_idx]

        fig.suptitle(clip_names[anim_idx], fontsize=16)
        lines.set_segments(skel_lines[frame])
        points._offsets3d = (skel[frame, :, 0], skel[frame, :, 1], skel[frame, :, 2])

        if cap is not None and image is not None:
            ret, frame_img = cap.read()
            if ret:
                frame_img = frame_img[:, :, [2, 1, 0]]
                image.set_data(frame_img)

    ani = FuncAnimation(fig, animate, interval=30)
    ani.resume()
    plt.show()


if __name__ == "__main__":
    main()
