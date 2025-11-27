import pickle
import pandas as pd
import numpy as np


SUBJECTS    = "csvs/subjects.csv"
CLIPS       = "csvs/clips.csv"
SKEL_ARRAYS = "output/all_124_nosub1_resized_skels.npz"

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


partitions = [
    ("output/p5/partition0.csv", "output/output4/part0.pkl"),
    ("output/p5/partition1.csv", "output/output4/part1.pkl"),
    ("output/p5/partition2.csv", "output/output4/part2.pkl"),
]

for partition_path, output_path in partitions:
    partition  = pd.read_csv(partition_path)
    out = partition_csv_to_skel_pkl(partition)

    print(f"Output to {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(out, f)
