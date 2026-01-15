#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def load_subjects(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def add_stability_pattern(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["stability_pattern"] = "mixed"
    out.loc[(out["stable"] > 0) & (out["unstable"] == 0), "stability_pattern"] = "all_stable"
    out.loc[(out["stable"] == 0) & (out["unstable"] > 0), "stability_pattern"] = "all_unstable"
    return out


def split_train_val(subset: pd.DataFrame, val_ratio: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts, val_parts = [], []
    rng_seed = seed

    for group in ["sarcopenia", "normal"]:
        for pattern in ["all_stable", "all_unstable", "mixed"]:
            part = subset[
                (subset["sarcopenia-normal"] == group) & (subset["stability_pattern"] == pattern)
            ]
            if len(part) == 0:
                continue
            if len(part) == 1:
                train_parts.append(part)
                continue

            val_size = max(1, int(round(val_ratio * len(part))))
            val_size = min(val_size, len(part) - 1)
            val_set = part.sample(n=val_size, random_state=rng_seed)
            train_set = part.drop(val_set.index)
            train_parts.append(train_set)
            val_parts.append(val_set)
            rng_seed += 1

    train = pd.concat(train_parts).reset_index(drop=True) if train_parts else pd.DataFrame()
    val = pd.concat(val_parts).reset_index(drop=True) if val_parts else pd.DataFrame()
    return train, val


def format_split(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, partition: int) -> pd.DataFrame:
    for df, split_name in ((train, "train"), (val, "val"), (test, "test")):
        df["split"] = split_name
        df["partition"] = partition
        if "stability_pattern" in df.columns:
            df.drop(columns=["stability_pattern"], inplace=True)

    combined = pd.concat([train, val, test]).reset_index(drop=True)
    return combined[
        [
            "subject",
            "sarcopenia-normal",
            "original_subject_idx",
            "clip_count",
            "stable",
            "unstable",
            "partition",
            "split",
        ]
    ]


def write_partition(df: pd.DataFrame, output_dir: str, partition_idx: int):
    prefix = f"partition{partition_idx}"
    csv_path = os.path.join(output_dir, f"{prefix}.csv")
    txt_path = os.path.join(output_dir, f"{prefix}.txt")
    df.to_csv(csv_path)
    with open(txt_path, "w", encoding="utf-8") as txt_file:
        txt_file.write(f"Partition {partition_idx}\n")
    print(f"Wrote {csv_path} and {txt_path}")


def main():
    parser = argparse.ArgumentParser(description="Double-stratified multi-partition subject split.")
    parser.add_argument("--subjects-csv", default="csvs/subjects.csv", help="Subjects metadata CSV.")
    parser.add_argument("--output-dir", default="output/double_stratified_partitions", help="Output directory.")
    parser.add_argument("--partitions", type=int, default=5, help="Number of partitions.")
    parser.add_argument("--train-ratio", type=float, default=0.65, help="Train ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.11, help="Validation ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.24, help="Test ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if not np.isclose(ratio_sum, 1.0, atol=1e-6):
        raise ValueError("train-ratio + val-ratio + test-ratio must sum to 1.0")
    if args.partitions < 1:
        raise ValueError("partitions must be >= 1")

    subjects = add_stability_pattern(load_subjects(args.subjects_csv))
    labels = subjects["sarcopenia-normal"] + "_" + subjects["stability_pattern"]

    os.makedirs(args.output_dir, exist_ok=True)
    seen_tests = set()
    nominal_test_size = max(1, int(round(args.test_ratio * len(subjects))))

    for partition_idx in range(1, args.partitions + 1):
        available = subjects[~subjects["subject"].isin(seen_tests)]
        available_labels = available["sarcopenia-normal"] + "_" + available["stability_pattern"]

        if len(available) == 0:
            raise RuntimeError("No subjects left to assign to test partitions without overlap.")

        test_size = min(nominal_test_size, len(available))
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=args.seed + partition_idx
        )
        trainval_idx, test_idx = next(splitter.split(available, available_labels))

        test = available.iloc[test_idx].copy()
        seen_tests.update(test["subject"])

        trainval = subjects[~subjects["subject"].isin(test["subject"])].copy()
        effective_val_ratio = args.val_ratio / (1.0 - args.test_ratio) if (1.0 - args.test_ratio) > 0 else 0
        train, val = split_train_val(trainval, effective_val_ratio, seed=args.seed + partition_idx)

        combined = format_split(train, val, test, partition=partition_idx)
        write_partition(combined, args.output_dir, partition_idx)


if __name__ == "__main__":
    main()
