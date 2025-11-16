import pandas as pd
from pathlib import Path

SUBJECTS   = Path("csvs/subjects.csv")
CLIPS      = Path("csvs/clips.csv")
FOLDS = [
    Path("folds_diverse/fold1_subjects.csv"),
    Path("folds_diverse/fold2_subjects.csv"),
    Path("folds_diverse/fold3_subjects.csv"),
    Path("folds_diverse/fold4_subjects.csv"),
]

subjects = pd.read_csv(SUBJECTS, index_col=0)
clips = pd.read_csv(CLIPS, index_col=0)

for fold_idx, fold in enumerate(FOLDS):
    print(f"{fold}:")
    df = pd.read_csv(fold)

    total_subj = len(df.index)
    total_clips = df["clip_count"].sum()
    for split in ["train", "val", "test"]:
        subj_count = (df["split"] == split).sum()
        sarcopenia_count = ((df["split"] == split) & (df["sarcopenia-normal"] == "sarcopenia")).sum()
        normal_count = ((df["split"] == split) & (df["sarcopenia-normal"] == "normal")).sum()
        total_subj_in_split = normal_count + sarcopenia_count
        unstable_clip_count = df.loc[df["split"] == split, "unstable"].sum()
        stable_clip_count = df.loc[df["split"] == split, "stable"].sum()
        total_clips_in_split = stable_clip_count + unstable_clip_count
        "For each fold’s train, validation, and test sets, the number of sarcopenia, normal, stable, and unstable cases"

        print(
            f"{split}: {subj_count} subjects ({subj_count / total_subj*100:.2f}% of all dataset), \n"
            f"         {sarcopenia_count} sarcopenia subjects ({sarcopenia_count / total_subj_in_split*100:.2f}%), "
            f"{normal_count} normal subjects ({normal_count / total_subj_in_split*100:.2f}%)  (% of all subjects in '{split}')\n"
            f"         {unstable_clip_count} unstable clips ({unstable_clip_count /total_clips_in_split*100:.2f}%), "
            f"{stable_clip_count} stable clips ({stable_clip_count / total_clips_in_split*100:.2f}%) (% of all clips in '{split}')\n"
        )

    if fold_idx == 3:
        print("Per-id information (original IDs)")
        for subj_idx, subj in df.iterrows():
            label = subj["sarcopenia-normal"]
            id = subj["original_subject_idx"]
            clips = subj["clip_count"]

            print(f"{label}_{id}: {clips} clips")

