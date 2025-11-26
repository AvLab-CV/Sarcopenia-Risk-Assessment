import pickle
import pandas as pd
import numpy as np


SUBJECTS    = "csvs/subjects.csv"
CLIPS       = "csvs/clips.csv"
SKEL_ARRAYS = "output/output2/skels.npz"

# import argparse
# args = argparse.ArgumentParser()
# args.add_argument('fold', help="fold CSV")
# args.add_argument('output', help="the output")
# args = args.parse_args()
# FOLD        = args.fold
# OUTPUT      = args.output

skels = np.load(SKEL_ARRAYS)
subjects = pd.read_csv(SUBJECTS, index_col=0)
clips = pd.read_csv(CLIPS, index_col=0)

def fold_csv_to_skel_pkl(fold):
    train_X = []
    train_Y = []
    train_clips = []
    val_X   = []
    val_Y   = []
    val_clips = []
    test_X  = []
    test_Y  = []
    test_clips = []

    for subj_idx, subj in fold.iterrows():
        subj_id = subjects.iloc[subj_idx]["subject"]
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


folds = [
    ("output/p5/partition0.csv", "output/output3/part0.pkl"),
    ("output/p5/partition1.csv", "output/output3/part1.pkl"),
    ("output/p5/partition2.csv", "output/output3/part2.pkl"),
]

for fold_path, output_path in folds:
    fold  = pd.read_csv(fold_path)
    out = fold_csv_to_skel_pkl(fold)

    print(f"Output to {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(out, f)
