import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Path3DCollection
import numpy as np
from matplotlib.animation import FuncAnimation
import pickle
import einops
import cv2
from pathlib import Path

PICKLE_PATH = "output2/NEW_video_skel.pkl"
PATHS_PATH = "output2/NEW_paths.txt"

VIDEO_PATH = Path("/Users/aldo/Code/avlab/dataset/resized_400/")
VIDEOS = [Path(p.strip()) for p in open(PATHS_PATH, "r")]
VIDEOS = [VIDEO_PATH / p.parts[-2] / p.parts[-1] for p in VIDEOS]

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


with open(PICKLE_PATH, 'rb') as f:
    skels = pickle.load(f)


# DEBUG: Only visualize first 100
skels = skels[:100]

anim_idx = 0
base_idx = 0

for i in range(len(skels)):
    skels[i] = einops.rearrange(skels[i], 'frame (joint coord) -> frame joint coord', coord=3)

    # # Feet on y=0 (every frame)
    # ground = (skels[i][:, 14] + skels[i][:, 18]) / 2
    # ground = einops.reduce(ground, "frame coord -> coord", reduction="mean")
    # skels[i] -= ground

    # # Calculate new average direction
    # # vec_up = (skels[i][:, 3] - (skels[i][:, 17] + skels[i][:, 13]) / 2)
    # vec_right = skels[i][-8:, 8] - skels[i][-8:, 4]
    # vec_up = skels[i][:, 2]

    # vec_right = vec_right.sum(axis=0)
    # vec_up = vec_up.sum(axis=0)

    # vec_fwd = np.cross(vec_right, vec_up)
    # rotation = np.column_stack([
    #     vec_right / np.linalg.norm(vec_right),
    #     vec_up / np.linalg.norm(vec_up),
    #     vec_fwd / np.linalg.norm(vec_fwd),
    # ])
    # # Orthogonalize
    # # rotation = np.linalg.qr(rotation)[0]

    # # Matrix multiplication
    # print(len(skels))
    # print(skels[0].shape)
    # skels[i] = einops.einsum(rotation, skels[i], "coord_i coord_j, frame joint coord_j -> frame joint coord_i")

    # Y <-> Z
    skels[i] = skels[i][:, :, [0, 2, 1]]
    # skels[i][:, :, 0] *= -1.0
    skels[i][:, :, 2] *= -1.0
    skels[i][:, :, 2] += 1

skels_lines = [skel[:, SKELETON_CONNECTIONS, :] for skel in skels]

# Video
path = VIDEOS[0]
cap = cv2.VideoCapture(str(path))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


fig = plt.figure()
ax: Axes3D = fig.add_subplot(1, 2, 1, projection='3d')
points: Path3DCollection = ax.scatter(*skels[anim_idx][0].T)  # Points

ax2: Axes3D = fig.add_subplot(1, 2, 2)
image = ax2.imshow(np.zeros((10,10,3)))

lines = Line3DCollection(skels_lines[anim_idx][0])
ax.add_collection3d(lines)

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(0, 2)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

def animate(i):
    global anim_idx, base_idx, cap, fps, frame_count
    
    frame = i - base_idx
    if frame >= skels[anim_idx].shape[0]:
        base_idx = i
        anim_idx += 1
        frame = 0
        print(f"Next animation: {anim_idx}")
        cap.release()
        path = VIDEOS[anim_idx]
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"FPS = {fps} framecount={frame_count} size={skels[anim_idx].shape[0]}")


    skel = skels[anim_idx]
    skel_lines = skels_lines[anim_idx]

    fig.suptitle(VIDEOS[anim_idx], fontsize=16)
    lines.set_segments(skel_lines[frame])
    points._offsets3d = (skel[frame, :, 0], skel[frame, :, 1], skel[frame, :, 2])

    ret, frame = cap.read()
    if ret:
        frame = frame[:, :, [2, 1, 0]]
        image.set_data(frame)

ani = FuncAnimation(fig, animate, interval=30)
ani.resume()
plt.show()

