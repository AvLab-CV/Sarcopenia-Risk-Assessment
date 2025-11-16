# Creates ZIP file with the entire dataset pre-split into folds according to CSVs in the FOLDS list

import io
import zipfile
import pandas as pd
from pathlib import Path

VIDEO_PATH = Path("/Users/aldo/Code/avlab/dataset/all_ver2")
SUBJECTS   = Path("csvs/subjects.csv")
CLIPS      = Path("csvs/clips.csv")
FOLDS = [
    Path("folds/fold1_subjects.csv"),
    Path("folds/fold2_subjects.csv"),
    Path("folds/fold3_subjects.csv"),
    Path("folds/fold4_subjects.csv"),
]
ZIP_OUT = "/Users/aldo/Downloads/folds_ver3.zip"

subjects = pd.read_csv(SUBJECTS, index_col=0)
clips = pd.read_csv(CLIPS, index_col=0)

zip_buffer = io.BytesIO()
with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zf:
    for fold_idx, fold in enumerate(FOLDS):
        df = pd.read_csv(fold)

        for subj_idx, subj in df.iterrows():
            subj_id = subjects.iloc[subj_idx]["subject"]
            subj_clips = clips.loc[clips["subject"] == subj_id]
            subj_clip_paths = subj_clips["clip_path"]
            subj_split = subj["split"]

            for path in subj_clip_paths:
                path_in = VIDEO_PATH / path
                path_out= f"fold{fold_idx+1}/{subj_split}/{path}"
                print(f"{path_in} -> {path_out}")
                zf.write(path_in, path_out)

        zf.write(fold, f"fold{fold_idx+1}.csv")

    zf.write(SUBJECTS, "subjects.csv")
    zf.write(CLIPS, "clips.csv")


with open(ZIP_OUT, 'wb') as f:
    f.write(zip_buffer.getvalue())
