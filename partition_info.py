import argparse
import io
from pathlib import Path

import pandas as pd


def print_to_string(*args, **kwargs):
    output = io.StringIO()
    print(*args, file=output, **kwargs)
    contents = output.getvalue()
    output.close()
    return contents

def get_partition_info(partition_paths) -> str:
    text = ""

    for partition_idx, partition_path in enumerate(partition_paths):
        text += print_to_string(f"{partition_path}:")
        df = pd.read_csv(partition_path)

        total_subj = len(df.index)
        for split in ["train", "val", "test"]:
            subj_count = (df["split"] == split).sum()
            sarcopenia_count = ((df["split"] == split) & (df["sarcopenia-normal"] == "sarcopenia")).sum()
            normal_count = ((df["split"] == split) & (df["sarcopenia-normal"] == "normal")).sum()
            total_subj_in_split = normal_count + sarcopenia_count
            unstable_clip_count = df.loc[df["split"] == split, "unstable"].sum()
            stable_clip_count = df.loc[df["split"] == split, "stable"].sum()
            total_clips_in_split = stable_clip_count + unstable_clip_count
            "For each partition’s train, validation, and test sets, the number of sarcopenia, normal, stable, and unstable cases"

            text += print_to_string(
                f"{split}: {subj_count} subjects ({subj_count / total_subj*100:.2f}% of all dataset), \n"
                f"         {sarcopenia_count} sarcopenia subjects ({sarcopenia_count / total_subj_in_split*100:.2f}%), "
                f"{normal_count} normal subjects ({normal_count / total_subj_in_split*100:.2f}%)  (% of all subjects in '{split}')\n"
                f"         {unstable_clip_count} unstable clips ({unstable_clip_count /total_clips_in_split*100:.2f}%), "
                f"{stable_clip_count} stable clips ({stable_clip_count / total_clips_in_split*100:.2f}%) (% of all clips in '{split}')\n"
            )

        if partition_idx == 3:
            text += print_to_string("Per-id information (original IDs)")
            for subj_idx, subj in df.iterrows():
                label = subj["sarcopenia-normal"]
                id = subj["original_subject_idx"]
                clips = subj["clip_count"]

                text += print_to_string(f"{label}_{id}: {clips} clips")

    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("partition_dir", type=Path)
    parser.add_argument("partition_count", type=int)
    args = parser.parse_args()
    PARTITIONS_COUNT = args.partition_count
    PARTITION_DIR = args.partition_dir
    print(f"Looking for {PARTITIONS_COUNT} partitions in `{PARTITION_DIR}`")
    partition_paths = [
        PARTITION_DIR / f"partition{partition + 1}.csv"
        for partition in range(PARTITIONS_COUNT)
    ]
    print(get_partition_info(partition_paths), end="")

