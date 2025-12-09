import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
from scipy.stats import chi2_contingency
from tqdm import tqdm

# DEFAULT_CSV = Path(
#     "/Users/aldo/Code/avlab/SkateFormer_synced/work_dir/20251210_0105/"
#     "partition1/sliding_window/sliding_window_windowsize64_stride8.csv"
# )
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


def plot_roc_and_hist(
    instability_rate: np.ndarray,
    labels: np.ndarray,
    out_path: Path = None,
    bins: int = 20,
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 4.5))

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
    ax2.hist(instability_rate[labels == 0], bins=bin_edges, alpha=0.6, label="No Sarcopenia", linewidth=0.4)
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


RISK_THRESHOLD_DEFAULT = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze sliding window instability predictions.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Input CSV with sliding window predictions.")
    parser.add_argument("--window-size", type=int, default=None, help="Window size (defaults to value encoded in filename).")
    parser.add_argument(
        "--base-stride",
        type=int,
        default=None,
        help="Stride used to generate the file (defaults to value encoded in filename).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Effective stride to evaluate (must be a multiple of base stride).",
    )
    parser.add_argument("--plots", type=Path, default=None, help="Directory to save per-subject plots.")
    parser.add_argument("--plot-kind", choices=["bars", "dots"], default="bars", help="Plot style.")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Frames per second for time conversion.")
    parser.add_argument(
        "--roc-hist-out",
        type=Path,
        default=None,
        help="Optional path to save combined ROC and instability-rate histogram.",
    )
    parser.add_argument(
        "--risk-threshold",
        type=float,
        default=RISK_THRESHOLD_DEFAULT,
        help="Threshold on instability rate to split risk groups.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df, window_size, base_stride = load_predictions(args.csv, args.window_size, args.base_stride)
    target_stride = args.stride or base_stride
    df = apply_stride(df, base_stride, target_stride)
    df_metrics = add_unstable_metrics(df)

    print(f"Window size: {window_size} | stride used in file: {base_stride} | effective stride: {target_stride}")

    labels = df_metrics["subject_has_sarcopenia"].astype(int).to_numpy()
    instability_rate = df_metrics["unstable_fraction"].to_numpy()
    _, roc_auc = plot_roc_and_hist(instability_rate, labels, out_path=args.roc_hist_out)
    print(f"ROC AUC (instability rate as predictor): {roc_auc:.3f}")
    overall_prev = labels.mean()
    print(f"Overall sarcopenia prevalence: {overall_prev:.3f}")

    mask_low = instability_rate < args.risk_threshold
    mask_high = instability_rate >= args.risk_threshold
    for name, mask in [
        (f"R < {args.risk_threshold}", mask_low),
        (f"R >= {args.risk_threshold}", mask_high),
    ]:
        count = mask.sum()
        if count == 0:
            print(f"{name}: no samples")
            continue
        prevalence = labels[mask].mean()
        print(f"{name}: n={count}, sarcopenia prevalence={prevalence:.3f}")

    # Odds ratio of sarcopenia for R >= threshold vs R < threshold
    n0, n1 = mask_low.sum(), mask_high.sum()
    if n0 > 0 and n1 > 0:
        p0 = labels[mask_low].mean()
        p1 = labels[mask_high].mean()
        if p0 in {0, 1} or p1 in {0, 1}:
            print("Odds ratio: undefined (zero or full prevalence in a group).")
        else:
            odds0 = p0 / (1 - p0)
            odds1 = p1 / (1 - p1)
            print(f"Odds ratio (R >= {args.risk_threshold} vs R < {args.risk_threshold}): {odds1 / odds0:.3f}")
    else:
        print("Odds ratio: insufficient samples in one of the groups.")

    # Chi-squared test comparing sarcopenia prevalence between risk groups
    if n0 > 0 and n1 > 0:
        pos_low = labels[mask_low].sum()
        neg_low = n0 - pos_low
        pos_high = labels[mask_high].sum()
        neg_high = n1 - pos_high
        contingency = np.array([[pos_low, neg_low], [pos_high, neg_high]])
        chi2, p_value, _, _ = chi2_contingency(contingency, correction=False)
        print(f"Chi-squared test p-value (risk groups): {p_value:.4g}")
    else:
        print("Chi-squared test: insufficient samples in one of the groups.")

    if args.plots:
        plot_all(
            df_metrics,
            window_size=window_size,
            fps=args.fps,
            out_dir=args.plots,
            kind=args.plot_kind,
        )


if __name__ == "__main__":
    main()
