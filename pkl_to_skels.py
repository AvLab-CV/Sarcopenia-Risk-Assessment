import numpy as np
import os
import pickle
from pathlib import Path

PICKLE_PATH = Path("output2/NEW_video_skel.pkl")
PATHS_PATH  = Path("output2/NEW_paths.txt")
OUTPUT_PATH = Path("output2/skels")

VIDEO_NAMES = [Path(p.strip()) for p in open(PATHS_PATH, "r")]
VIDEO_NAMES = [p.parts[-1] for p in VIDEO_NAMES]

with open(PICKLE_PATH, 'rb') as f:
    skels = pickle.load(f)

# os.makedirs(OUTPUT_PATH, exist_ok=True)

arrays = {}
for i in range(len(skels)):
    # out = OUTPUT_PATH / (VIDEO_NAMES[i] + ".npy")
    name = VIDEO_NAMES[i]
    arrays[name] = skels[i]

np.savez("output2/skels.npz", **arrays)
