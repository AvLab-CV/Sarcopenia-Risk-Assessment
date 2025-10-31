import pickle
import random
import pandas as pd
import numpy as np

SUBJECTS    = "csvs/subjects.csv"
CLIPS       = "csvs/clips.csv"
FOLD        = "folds/fold1_subjects.csv"
SKEL_ARRAYS = "output2/skels.npz"
OUTPUT      = "output2/fold1.pkl"

skels = np.load(SKEL_ARRAYS)
subjects = pd.read_csv(SUBJECTS, index_col=0)
clips = pd.read_csv(CLIPS, index_col=0)
fold  = pd.read_csv(FOLD)

train_X = []
train_Y = []
val_X   = []
val_Y   = []
test_X  = []
test_Y  = []

for subj_idx, subj in fold.iterrows():
    subj_id = subjects.iloc[subj_idx]["subject"]
    subj_clips = clips.loc[clips["subject"] == subj_id]
    subj_skels = [skels[name] for name in subj_clips["clip_path"]]
    subj_labels = [int(stable_unstable == "unstable") for stable_unstable in subj_clips["stable-unstable"]]

    if subj["split"] == "train":
        train_X.extend(subj_skels)
        train_Y.extend(subj_labels)
    if subj["split"] == "val":
        val_X.extend(subj_skels)
        val_Y.extend(subj_labels)
    if subj["split"] == "test":
        test_X.extend(subj_skels)
        test_Y.extend(subj_labels)

out = dict(
    train_X=train_X,
    train_Y=train_Y,
    val_X=val_X,
    val_Y=val_Y,
    test_X=test_X,
    test_Y=test_Y,
)

print(f"Output to {OUTPUT}")
with open(OUTPUT, 'wb') as f:
    pickle.dump(out, f)
