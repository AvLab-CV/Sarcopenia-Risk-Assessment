import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from seq_transformation import seq_transformation

SUBJECTS    = "csvs/subjects.csv"
CLIPS       = "csvs/clips.csv"
SKEL_ARRAYS = "output/skeletons/skels_all_124_nosub1_resized.npz"

skels = np.load(SKEL_ARRAYS)
subjects = pd.read_csv(SUBJECTS, index_col=0)
clips = pd.read_csv(CLIPS, index_col=0)

def partition_csv_to_skel_pkl(partition):
    train_X = []
    train_Y = []
    train_clips = []
    val_X   = []
    val_Y   = []
    val_clips = []
    test_X  = []
    test_Y  = []
    test_clips = []

    for subj_idx, subj in partition.iterrows():
        subj_id = subj["subject"]
        subj_clips = clips.loc[clips["subject"] == subj_id]
        subj_clip_paths = subj_clips["clip_path"]
        subj_skels = [skels[name] for name in subj_clip_paths]
        subj_labels = [int(stable_unstable == "unstable") for stable_unstable in subj_clips["stable-unstable"]]

        if subj["split"] == "train":
            train_X.extend(subj_skels)
            train_Y.extend(subj_labels)
            train_clips.extend(subj_clip_paths)
        if subj["split"] == "val":
            val_X.extend(subj_skels)
            val_Y.extend(subj_labels)
            val_clips.extend(subj_clip_paths)
        if subj["split"] == "test":
            test_X.extend(subj_skels)
            test_Y.extend(subj_labels)
            test_clips.extend(subj_clip_paths)

    return dict(
        train_X=train_X,
        train_Y=train_Y,
        train_clips=train_clips,
        val_X=val_X,
        val_Y=val_Y,
        val_clips=val_clips,
        test_X=test_X,
        test_Y=test_Y,
        test_clips=test_clips,
    )



parser = argparse.ArgumentParser()
parser.add_argument(
    "input_dir",
    type=Path,
    help="Directory containing the outputs to process.",
)
args = parser.parse_args()

# INPUT_DIR  = Path("output/20251128_1753_partition")
INPUT_DIR  = args.input_dir
# OUTPUT_DIR = INPUT_DIR.parent / (INPUT_DIR.parts[-1] + "_skel")
OUTPUT_DIR = INPUT_DIR
inputs = [p for p in INPUT_DIR.iterdir() if p.suffix == ".csv"]
inputs.sort()

# print("Using:\n -", '\n - '.join(str(x) for x in inputs))
print("Assigning skeletons + performing seq transformation")
for input_path in inputs:

    partition  = pd.read_csv(input_path)
    pkl = partition_csv_to_skel_pkl(partition)
    seq_npz = seq_transformation(pkl)

    OUTPUT_DIR.mkdir(exist_ok=True)

    # output_path_pkl = OUTPUT_DIR / (input_path.stem + ".pkl")
    # print(f"{input_path} -> {output_path_pkl}")
    # with open(output_path_pkl, 'wb') as f:
    #     pickle.dump(pkl, f)

    output_path_seq = OUTPUT_DIR / (input_path.stem + ".npz")
    print(f"{input_path} -> {output_path_seq}")
    np.savez(output_path_seq, **seq_npz)
