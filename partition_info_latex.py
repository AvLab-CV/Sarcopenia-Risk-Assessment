#!/usr/bin/env python3
"""
Generate LaTeX tables describing how subjects and clips are distributed
across a set of cross-validation partitions. Inspired by partition_info.py.
"""

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

SPLITS: Iterable[str] = ("train", "val", "test")


def load_partitions(partition_dir: Path, count: int) -> List[pd.DataFrame]:
    """Load partition CSVs named partition1.csv, partition2.csv, ..."""
    partition_paths = [
        partition_dir / f"partition{idx + 1}.csv" for idx in range(count)
    ]
    missing = [p for p in partition_paths if not p.exists()]
    if missing:
        missing_list = ", ".join(str(m) for m in missing)
        raise FileNotFoundError(f"Missing partition files: {missing_list}")
    return [pd.read_csv(p) for p in partition_paths]


def summarize_split(df: pd.DataFrame, split: str) -> Dict[str, int]:
    subset = df[df["split"] == split]
    sarcopenia = int((subset["sarcopenia-normal"] == "sarcopenia").sum())
    normal = int((subset["sarcopenia-normal"] == "normal").sum())
    unstable_clips = int(subset["unstable"].sum())
    stable_clips = int(subset["stable"].sum())
    return {
        "subjects": len(subset),
        "sarcopenia": sarcopenia,
        "normal": normal,
        "unstable_clips": unstable_clips,
        "stable_clips": stable_clips,
        "total_clips": unstable_clips + stable_clips,
    }


def build_subject_table(partitions: List[pd.DataFrame], fold_offset: int = 0) -> str:
    rows = []
    for idx, df in enumerate(partitions):
        split_stats = {split: summarize_split(df, split) for split in SPLITS}
        row = (
            f"Fold {idx + fold_offset} & "
            f"{split_stats['train']['subjects']} ({split_stats['train']['sarcopenia']} S / {split_stats['train']['normal']} N) & "
            f"{split_stats['val']['subjects']} ({split_stats['val']['sarcopenia']} S / {split_stats['val']['normal']} N) & "
            f"{split_stats['test']['subjects']} ({split_stats['test']['sarcopenia']} S / {split_stats['test']['normal']} N) \\\\"
        )
        rows.append(row)

    table = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Subject Distribution Across 3-Fold Cross-Validation.\\",
        r"S: Sarcopenia, N: Normal.}",
        r"\label{tab:dataset_partitions}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Partition} & \textbf{Train} & \textbf{Val} & \textbf{Test} \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(table)


def build_clip_table(partitions: List[pd.DataFrame], fold_offset: int = 0) -> str:
    rows = []
    for idx, df in enumerate(partitions):
        split_stats = {split: summarize_split(df, split) for split in SPLITS}
        row = (
            f"Fold {idx + fold_offset} & "
            f"{split_stats['train']['total_clips']} ({split_stats['train']['unstable_clips']} U / {split_stats['train']['stable_clips']} S) & "
            f"{split_stats['val']['total_clips']} ({split_stats['val']['unstable_clips']} U / {split_stats['val']['stable_clips']} S) & "
            f"{split_stats['test']['total_clips']} ({split_stats['test']['unstable_clips']} U / {split_stats['test']['stable_clips']} S) \\\\"
        )
        rows.append(row)

    table = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Clip Distribution Across 3-Fold Cross-Validation.\\",
        r"U: Unstable, S: Stable.}",
        r"\label{tab:clips_partition}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Partition} & \textbf{Train Clips} & \textbf{Val Clips} & \textbf{Test Clips} \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(table)


def main(partition_dir: Path, partition_count: int, fold_offset: int) -> None:
    partitions = load_partitions(partition_dir, partition_count)
    subject_table = build_subject_table(partitions, fold_offset)
    clip_table = build_clip_table(partitions, fold_offset)
    print(subject_table)
    print()  # spacer between tables
    print(clip_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables summarizing partition subject and clip counts."
    )
    parser.add_argument("partition_dir", type=Path, help="Directory containing partition CSVs.")
    parser.add_argument(
        "partition_count",
        type=int,
        nargs="?",
        default=3,
        help="Number of partitions to load (default: 3).",
    )
    parser.add_argument(
        "--fold-offset",
        type=int,
        default=0,
        help="Offset for fold numbering in the table (default: 0).",
    )
    args = parser.parse_args()
    main(args.partition_dir, args.partition_count, args.fold_offset)
