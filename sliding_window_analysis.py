import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
from scipy.stats import chi2_contingency, ttest_ind
from tqdm import tqdm

DEFAULT_CSV = Path()
DEFAULT_FPS = 30


def parse_window_and_stride_from_name(csv_path: Path):
    match = re.search(r"windowsize(\d+)_stride(\d+)", csv_path.name)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def load_predictions(csv_path: Path, window_size: int, stride: int):
    inferred_window, inferred_stride = parse_window_and_stride_from_name(csv_path)
    window_size = window_size or inferred_window
    stride = stride or inferred_stride
    if window_size is None or stride is None:
        raise ValueError("window_size and stride must be provided or encoded in the filename.")

    df = pd.read_csv(csv_path)
    df["predictions"] = df["predictions"].map(lambda x: np.fromstring(str(x).strip("[]"), sep=" "))
    return df, window_size, stride


def apply_stride(df: pd.DataFrame, base_stride: int, target_stride: int) -> pd.DataFrame:
    if target_stride % base_stride != 0:
        raise ValueError("target stride must be a multiple of the stride used to create the file.")
    step = target_stride // base_stride
    preds = df["predictions"] if step == 1 else df["predictions"].map(lambda x: x[::step])
    window_pos = preds.map(lambda x: np.arange(x.shape[0]) * target_stride)
    return df.assign(predictions=preds, window_pos=window_pos)


def add_unstable_metrics(
    df: pd.DataFrame
) -> pd.DataFrame:
    stable_count = df["predictions"].map(lambda x: (x <= 0.5).sum())
    unstable_count = df["predictions"].map(lambda x: (x > 0.5).sum())
    unstable_fraction = unstable_count / (stable_count + unstable_count)
    return df.assign(
        stable_count=stable_count,
        unstable_count=unstable_count,
        unstable_fraction=unstable_fraction,
    )


def load_subject_mapping(mapping_csv: Path) -> dict[int, int]:
    mapping_df = pd.read_csv(mapping_csv)
    if not {"index", "subject"}.issubset(mapping_df.columns):
        raise ValueError("Mapping CSV must have 'index' and 'subject' columns.")
    return dict(zip(mapping_df["index"], mapping_df["subject"]))


def add_subject_ids(df: pd.DataFrame, index_to_subject: dict[int, int]) -> pd.DataFrame:
    subjects = df.index.map(index_to_subject.get)
    if subjects.isna().any():
        raise ValueError("Some sliding-window rows are missing a subject mapping.")
    return df.assign(subject=subjects.astype(int))


def load_partition_subjects(partition_csv: Path, split: str) -> set[int]:
    part_df = pd.read_csv(partition_csv)
    if "split" not in part_df.columns or "subject" not in part_df.columns:
        raise ValueError("Partition CSV must include 'subject' and 'split' columns.")
    return set(part_df.loc[part_df["split"] == split, "subject"].astype(int).tolist())


def plot_roc_and_hist(
    instability_rate: np.ndarray,
    labels: np.ndarray,
    out_path: Path = None,
    bins: int = 20,
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3.0))

    fpr, tpr, _ = roc_curve(labels, instability_rate)
    roc_auc = auc(fpr, tpr)

    ax1.plot(fpr, tpr, linewidth=1.8, label=f"AUC = {roc_auc:.3f}")
    ax1.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.legend(frameon=False)

    bin_edges = np.linspace(0, 1, bins + 1)
    ax2.hist(instability_rate[labels == 0], bins=bin_edges, alpha=0.6, label="Normal", linewidth=0.4)
    ax2.hist(instability_rate[labels == 1], bins=bin_edges, alpha=0.6, label="Sarcopenia", linewidth=0.4)
    ax2.set_xlabel("Instability Rate")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution by Sarcopenia Status")
    ax2.legend(frameon=False)

    fig.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", dpi=300)
    return fig, roc_auc


def plot_subject(df: pd.DataFrame, idx: int, window_size: int, fps: int, kind: str = "bars"):
    row = df.iloc[idx]
    predictions: np.ndarray = row["predictions"]
    window_pos = row["window_pos"].astype(np.float32) + (window_size / 2)
    window_pos /= float(fps)
    seq_len = float(row["seq_len"]) / float(fps)
    threshold = 0.5

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(0.0, seq_len)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("P(unstable | window)")

    mask = predictions > threshold
    ax.plot(window_pos, predictions, color="gray", linewidth=1)
    ax.axhline(threshold, alpha=0.6, color=(0, 0.5, 0), linestyle="--", linewidth=1.5, label="unstable threshold")
    if kind == "dots":
        ax.scatter(window_pos[~mask], predictions[~mask], color="blue", label="stable")
        ax.scatter(window_pos[mask], predictions[mask], color="red", label="unstable")
    else:
        ax.bar(window_pos[~mask], predictions[~mask], width=window_size / fps, color="blue", alpha=0.15)
        ax.scatter(window_pos[~mask], predictions[~mask], color="blue", label="stable")
        ax.bar(window_pos[mask], predictions[mask], width=window_size / fps, color="red", alpha=0.15)
        ax.scatter(window_pos[mask], predictions[mask], color="red", label="unstable")

    ax.legend()
    ax.grid()
    fig.tight_layout()
    return fig


def plot_all(df: pd.DataFrame, window_size: int, fps: int, out_dir: Path, kind: str = "bars"):
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx in tqdm(range(len(df.index)), desc="Plotting"):
        fig = plot_subject(df, idx, window_size, fps, kind=kind)
        fig.savefig(out_dir / f"{idx}_stability_plot.pdf")
        fig.clear()
        plt.close(fig)


def print_metrics_report(
    name: str,
    labels: np.ndarray,
    instability_rate: np.ndarray,
    risk_threshold: float,
    roc_hist_out: Path = None,
):
    print(f"\n=== {name} ===")
    _, roc_auc = plot_roc_and_hist(instability_rate, labels, out_path=roc_hist_out)
    print(f"ROC AUC (instability rate as predictor): {roc_auc:.3f}")
    overall_pos = labels.sum()
    overall_prev = overall_pos / len(labels)
    print(f"Overall sarcopenia prevalence: {overall_prev:.3f} (count={overall_pos})")

    sarc_vals = instability_rate[labels == 1]
    normal_vals = instability_rate[labels == 0]
    if sarc_vals.size > 0 and normal_vals.size > 0:
        sarc_mean, sarc_std = sarc_vals.mean(), sarc_vals.std(ddof=1)
        normal_mean, normal_std = normal_vals.mean(), normal_vals.std(ddof=1)
        t_stat, p_val = ttest_ind(sarc_vals, normal_vals, equal_var=False)
        print("Instability rate by sarcopenia status:")
        print(f"  Sarcopenia: mean R = {sarc_mean:.2f} ± {sarc_std:.2f} (n={len(sarc_vals)})")
        print(f"  Normal:     mean R = {normal_mean:.2f} ± {normal_std:.2f} (n={len(normal_vals)})")
        print(f"  Two-sample t-test (Welch): t = {t_stat:.2f}, p = {p_val:.3f}")
        print(
            "  Interpretation: sarcopenic subjects showed higher instability rates "
            "than normal subjects."
        )
    else:
        print("Group stats/t-test: insufficient samples in one or both groups.")

    mask_low = instability_rate < risk_threshold
    mask_high = instability_rate >= risk_threshold
    for group_name, mask in [
        (f"R < {risk_threshold}", mask_low),
        (f"R >= {risk_threshold}", mask_high),
    ]:
        count = mask.sum()
        if count == 0:
            print(f"{group_name}: no samples")
            continue
        pos = labels[mask].sum()
        prevalence = pos / count
        print(f"{group_name}: n={count}, sarcopenia prevalence={prevalence:.3f}, count={pos}")

    # Odds ratio of sarcopenia for R >= threshold vs R < threshold
    n0, n1 = mask_low.sum(), mask_high.sum()
    if n0 > 0 and n1 > 0:
        pos_low = labels[mask_low].sum()
        neg_low = n0 - pos_low
        pos_high = labels[mask_high].sum()
        neg_high = n1 - pos_high
        p0 = pos_low / n0
        p1 = pos_high / n1
        if p0 in {0, 1} or p1 in {0, 1}:
            print("Odds ratio: undefined (zero or full prevalence in a group).")
        else:
            odds0 = p0 / (1 - p0)
            odds1 = p1 / (1 - p1)
            or_value = odds1 / odds0
            se_log_or = np.sqrt(1 / pos_high + 1 / neg_high + 1 / pos_low + 1 / neg_low)
            ci_low = np.exp(np.log(or_value) - 1.96 * se_log_or)
            ci_high = np.exp(np.log(or_value) + 1.96 * se_log_or)
            print(
                f"Odds ratio (R >= {risk_threshold} vs R < {risk_threshold}): "
                f"{or_value:.3f} (SE={se_log_or:.3f}, 95% CI=[{ci_low:.3f}, {ci_high:.3f}])"
            )
    else:
        print("Odds ratio: insufficient samples in one of the groups.")

    # Chi-squared test comparing sarcopenia prevalence between risk groups
    if n0 > 0 and n1 > 0:
        contingency = np.array(
            [
                [labels[mask_low].sum(), n0 - labels[mask_low].sum()],
                [labels[mask_high].sum(), n1 - labels[mask_high].sum()],
            ]
        )
        chi2, p_value, _, _ = chi2_contingency(contingency, correction=False)
        print(f"Chi-squared test p-value (risk groups): {p_value:.4g}")
    else:
        print("Chi-squared test: insufficient samples in one of the groups.")


def main():
    mapping_csv = Path("csvs/sliding_window_subject_idx_to_subject_id.csv")
    sliding_window_csvs = [
        Path("/Users/aldo/Code/avlab/SkateFormer_synced/work_dir/20251210_0105/partition1/sliding_window/sliding_window_windowsize64_stride8.csv"),
        Path("/Users/aldo/Code/avlab/SkateFormer_synced/work_dir/20251210_0105/partition2/sliding_window/sliding_window_windowsize64_stride8.csv"),
        Path("/Users/aldo/Code/avlab/SkateFormer_synced/work_dir/20251210_0105/partition3/sliding_window/sliding_window_windowsize64_stride8.csv"),
    ]
    partition_csvs = [
        Path("output/partitions/20251209_1738_partition/partition1.csv"),
        Path("output/partitions/20251209_1738_partition/partition2.csv"),
        Path("output/partitions/20251209_1738_partition/partition3.csv"),
    ]
    target_stride = 64  # e.g., 24 to subsample if the file stride is 8; None keeps original
    risk_threshold = 0.5
    roc_hist_out = Path("output/plots/risk_dist.pdf")
    plots_out = None  # e.g., Path("plots/")
    plot_kind = "bars"
    fps = DEFAULT_FPS
    split_name = "test"  # "test" or "val"
    # -------------------------------------------

    if len(sliding_window_csvs) != len(partition_csvs):
        raise ValueError("sliding_window_csvs and partition_csvs must have the same length.")

    index_to_subject = load_subject_mapping(mapping_csv)

    filtered_dfs: list[pd.DataFrame] = []
    window_sizes: set[int] = set()
    base_strides: set[int] = set()

    for fold_idx, (sw_csv, part_csv) in enumerate(zip(sliding_window_csvs, partition_csvs), start=1):
        df, window_size, base_stride = load_predictions(sw_csv, window_size=None, stride=None)
        window_sizes.add(window_size)
        base_strides.add(base_stride)

        effective_stride = target_stride or base_stride
        df = apply_stride(df, base_stride, effective_stride)
        df = add_subject_ids(df, index_to_subject)

        subjects_in_split = load_partition_subjects(part_csv, split=split_name)
        df = df[df["subject"].isin(subjects_in_split)].copy()
        filtered_dfs.append(df)

        if not df.empty:
            df_fold_metrics = add_unstable_metrics(df)
            labels_fold = df_fold_metrics["subject_has_sarcopenia"].astype(int).to_numpy()
            instability_fold = df_fold_metrics["unstable_fraction"].to_numpy()
            print_metrics_report(
                name=f"Fold {fold_idx}",
                labels=labels_fold,
                instability_rate=instability_fold,
                risk_threshold=risk_threshold,
                roc_hist_out=None,
            )

    if not filtered_dfs:
        print("No data after filtering partitions.")
        return

    df_all = pd.concat(filtered_dfs, ignore_index=True)
    df_metrics = add_unstable_metrics(df_all)

    window_desc = ", ".join(map(str, sorted(window_sizes))) or "unknown"
    stride_desc = ", ".join(map(str, sorted(base_strides))) or "unknown"
    print(f"Window sizes present: {window_desc}")
    print(f"Base strides present: {stride_desc}")
    print(f"Effective stride used: {target_stride or 'original'}")

    labels = df_metrics["subject_has_sarcopenia"].astype(int).to_numpy()
    instability_rate = df_metrics["unstable_fraction"].to_numpy()
    print_metrics_report(
        name="Aggregate",
        labels=labels,
        instability_rate=instability_rate,
        risk_threshold=risk_threshold,
        roc_hist_out=roc_hist_out,
    )

    if plots_out:
        window_for_plot = next(iter(window_sizes)) if window_sizes else None
        if window_for_plot is None:
            print("Plots requested but window size is unknown.")
        else:
            plot_all(
                df_metrics,
                window_size=window_for_plot,
                fps=fps,
                out_dir=plots_out,
                kind=plot_kind,
            )


if __name__ == "__main__":
    main()
