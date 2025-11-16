import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SUBJECTS   = Path("csvs/subjects.csv")
CLIPS      = Path("csvs/clips.csv")
FOLDS = [
    Path("folds/fold1_subjects.csv"),
    Path("folds/fold2_subjects.csv"),
    Path("folds/fold3_subjects.csv"),
    Path("folds/fold4_subjects.csv"),
]

subjects = pd.read_csv(SUBJECTS, index_col=0)
clips = pd.read_csv(CLIPS, index_col=0)
data = dict(
    train=dict(x=[], y=[], c='red'),
    val=dict(x=[], y=[], c='green'),
    test=dict(x=[], y=[], c='blue'),
)

for fold_idx, fold in enumerate(FOLDS):
    df = pd.read_csv(fold)
    for sidx, split in enumerate(["train", "val", "test"]):
        x = np.array(df.loc[df["split"] == split].index.tolist())
        y = np.ones_like(x) * fold_idx
        data[split]["x"].extend(x)
        data[split]["y"].extend(y)
# Plot
for split in ["train", "val", "test"]:
    x = data[split]["x"]
    y = data[split]["y"]
    c = data[split]["c"]
    plt.scatter(x, y, color=c)
plt.show()

# folds = [
#     [1, 4, 6, 7, 8, 14, 19, 20, 21, 25, 27, 29, 48, 60, 63, 75, 88, 90, 103, 108, 112, 118, 123, 129, 132, 136, 138],
#     [16, 18, 25, 26, 35, 49, 51, 54, 60, 65, 73, 78, 92, 101, 102, 112, 116, 117, 119, 128, 135, 139],
#     [4, 7, 16, 21, 26, 30, 35, 40, 41, 50, 57, 62, 65, 68, 71, 82, 84, 85, 95, 101, 112, 117, 124, 125, 130, 131, 134, 139],
#     [0, 24, 37, 40, 43, 44, 45, 49, 52, 59, 61, 66, 67, 69, 70, 74, 77, 90, 92, 94, 100, 101, 108, 116, 118, 121, 137, 138]
# ]

# # Change subject IDX to match order of appearance in test set. This just makes the plot cleaner
# # and easier to understand.
# counter = 0
# old_to_new_idx = {}
# new_folds = []
# for fold in folds:
#     new_fold = []
#     for x in fold:
#         if x not in old_to_new_idx:
#             old_to_new_idx[x] = counter
#             counter += 1
#         new_fold.append(old_to_new_idx[x])
#     new_folds.append(new_fold)
# folds = new_folds

