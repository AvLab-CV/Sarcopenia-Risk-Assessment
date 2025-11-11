import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Path3DCollection
import numpy as np
from matplotlib.animation import FuncAnimation
import pickle
import einops
import cv2
from pathlib import Path
import pandas as pd

# Base dir where containing all videos
VIDEO_DIR_PATH = Path("/Users/aldo/Code/avlab/dataset/resized_400_flat/")
# We need access to the skeletons, and to the path clips.
# This pickle contains it all. It is a dict.
# You can check which fields it has by seeing the `train_test_split.py` script
# in poseops.
FOLD_PKL_PATH = Path("/Users/aldo/Code/avlab/poseopsfinal/output2/fold1.pkl")
SET = "test" # in [train,val,test]

with open(FOLD_PKL_PATH, 'rb') as f:
    fold = pickle.load(f)

    skels = fold[SET + "_X"]
    labels = fold[SET + "_Y"]
    clip_paths = [VIDEO_DIR_PATH / p for p in fold[SET + "_clips"]]
    print(f"Total skeletons: {len(skels)}")

# Select skeletons to visualize
# OPTION: simple range
# indexes = list(range(2))

# OPTION: error analysis
wrong = pd.read_csv("/Users/aldo/Code/avlab/SkateFormer_synced/work_dir/fold1_test/test_wrong.txt")
wrong['index'] = wrong['index'].astype(np.uint32)
wrong['pred'] = ['unstable' if x == 1 else 'stable' for x in wrong['pred']]
wrong['gt'] = ['unstable' if x == 1 else 'stable' for x in wrong['gt']]
# wrong = wrong.sort_values('chosen_confidence')
print("Failed samples:")
print(wrong)
indexes = list(wrong['index'])

# HERE: we filter which skeletons we actually want to visualize (by indexes)
skels = [skels[i] for i in indexes]
labels = [labels[i] for i in indexes]
full_clip_paths = [clip_paths[i] for i in indexes]
titles = [
    f"Sample {row['index']} Pred={row['pred']} GT={row['gt']} conf={row['chosen_confidence']:.4f}"
    for _, row in wrong.iterrows()
]
N = len(skels)
print(f"Visualizing {N} skeletons")

anim_idx = 0
base_idx = 0

for i in range(N):
    skels[i] = einops.rearrange(skels[i], 'frame (joint coord) -> frame joint coord', coord=3)
    # Y <-> Z
    skels[i] = skels[i][:, :, [0, 2, 1]]
    # skels[i][:, :, 0] *= -1.0
    skels[i][:, :, 2] *= -1.0
    skels[i][:, :, 2] += 1

SKELETON_CONNECTIONS = np.array([
    # Torso
    [0, 1], [1, 20], [20, 2], [2, 3],
    # Left Arm
    [2, 4], [4, 5], [5, 6], [6, 7], [7, 21], [7, 22],
    # Right Arm
    [2, 8], [8, 9], [9, 10], [10, 11], [11, 23], [11, 24],
    # Left Leg
    [0, 12], [12, 13], [13, 14], [14, 15],
    # Right Leg
    [0, 16], [16, 17], [17, 18], [18, 19],
])
skels_lines = [skel[:, SKELETON_CONNECTIONS, :] for skel in skels]

# Plot stuff
fig = plt.figure()
fig.suptitle(titles[anim_idx], fontsize=16)
ax: Axes3D = fig.add_subplot(1, 2, 1, projection='3d')
points: Path3DCollection = ax.scatter(*skels[anim_idx][0].T)  # Points
lines = Line3DCollection(skels_lines[anim_idx][0])
ax.add_collection3d(lines)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(0, 2)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax2: Axes3D = fig.add_subplot(1, 2, 2)
image = ax2.imshow(np.zeros((10,10,3)))

# Video
print(f"Playing: {anim_idx}/{N}")
path = full_clip_paths[anim_idx]
cap = cv2.VideoCapture(str(path))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

def animate(i):
    global anim_idx, base_idx, cap, fps, frame_count
    if anim_idx >= N:
        return
    
    frame = i - base_idx
    if frame >= skels[anim_idx].shape[0]:
        base_idx = i
        anim_idx += 1
        cap.release()
        if anim_idx < N:
            print(f"Playing: {anim_idx}/{N}")
            path = full_clip_paths[anim_idx]
            cap = cv2.VideoCapture(str(path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # print(f"FPS = {fps} framecount={frame_count} size={skels[anim_idx].shape[0]}")
            frame = 0
        else:
            # ani.pause()
            # plt.close()
            # exit(0)
            # return
            
            # Reset
            print("Restarting")
            anim_idx = 0
            print(f"Playing: {anim_idx}/{N}")
            path = full_clip_paths[anim_idx]
            cap = cv2.VideoCapture(str(path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # print(f"FPS = {fps} framecount={frame_count} size={skels[anim_idx].shape[0]}")
            frame = 0

    skel = skels[anim_idx]
    skel_lines = skels_lines[anim_idx]

    fig.suptitle(titles[anim_idx], fontsize=16)
    lines.set_segments(skel_lines[frame])
    points._offsets3d = (skel[frame, :, 0], skel[frame, :, 1], skel[frame, :, 2])

    ret, cap_frame = cap.read()
    if ret:
        cap_frame = cap_frame[:, :, [2, 1, 0]]
        image.set_data(cap_frame)

# total_frames = sum(skel.shape[0] for skel in skels)
# print(f"{total_frames=}")
ani = FuncAnimation(fig, animate, interval=30, cache_frame_data=False)
ani.resume()
plt.show()
