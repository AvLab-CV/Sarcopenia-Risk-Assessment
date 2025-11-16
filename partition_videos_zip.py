# Creates ZIP file with the entire dataset pre-split into folds according to CSVs in the FOLDS list
import argparse
import io
import zipfile
from pathlib import Path

import pandas as pd

import partition_info

VIDEO_PATH = Path("/Users/aldo/Code/avlab/dataset/all_ver2")
SUBJECTS   = Path("csvs/subjects.csv")
CLIPS      = Path("csvs/clips.csv")
subjects = pd.read_csv(SUBJECTS, index_col=0)
clips = pd.read_csv(CLIPS, index_col=0)


parser = argparse.ArgumentParser()
parser.add_argument("partition_dir", type=Path)
parser.add_argument("partition_count", type=int)
parser.add_argument("zip_out", type=Path)
args = parser.parse_args()
PARTITIONS_COUNT = args.partition_count
PARTITION_DIR = args.partition_dir
print(f"Looking for {PARTITIONS_COUNT} partitions in `{PARTITION_DIR}`")
partition_paths = [
    PARTITION_DIR / f"partition{partition + 1}.csv"
    for partition in range(PARTITIONS_COUNT)
]
ZIP_OUT = args.zip_out

zip_buffer = io.BytesIO()
with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zf:
    for partition_idx, partition in enumerate(partition_paths):
        df = pd.read_csv(partition)

        for subj_idx, subj in df.iterrows():
            subj_id = subjects.iloc[subj_idx]["subject"]
            subj_clips = clips.loc[clips["subject"] == subj_id]
            subj_clip_paths = subj_clips["clip_path"]
            subj_split = subj["split"]

            for path in subj_clip_paths:
                path_in = VIDEO_PATH / path
                path_out= f"partition{partition_idx+1}/{subj_split}/{path}"
                print(f"{path_in} -> {path_out}")
                zf.write(path_in, path_out)

        zf.write(partition, f"partition{partition_idx+1}.csv")

    zf.write(SUBJECTS, "subjects.csv")
    zf.write(CLIPS, "clips.csv")
    zf.writestr("info.txt", partition_info.get_partition_info(partition_paths))


with open(ZIP_OUT, 'wb') as f:
    f.write(zip_buffer.getvalue())
