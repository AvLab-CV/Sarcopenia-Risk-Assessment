import numpy as np
import pickle
from pathlib import Path

PICKLE_PATH = Path("output_merged/NEW_video_skel.pkl")
PATHS_PATH  = Path("output_merged/NEW_paths.txt")
OUTPUT_PATH = Path("output_merged/skels.npz")

VIDEO_NAMES = [Path(p.strip()) for p in open(PATHS_PATH, "r")]
VIDEO_NAMES = [p.parts[-1] for p in VIDEO_NAMES]

with open(PICKLE_PATH, 'rb') as f:
    skels = pickle.load(f)

arrays = {}
for i in range(len(skels)):
    name = VIDEO_NAMES[i]
    arrays[name] = skels[i]

np.savez(OUTPUT_PATH, **arrays)
