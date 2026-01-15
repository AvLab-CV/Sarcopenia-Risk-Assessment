import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tensorboard.backend.event_processing import event_accumulator

WORK_DIR = Path("/Users/aldo/Code/avlab/SkateFormer_synced/work_dir")
LOG_SUBDIR = Path("runs/train")
REPORT_ROOT = Path("output/reports")
PLOTS_ROOT = Path("output/plots")

@dataclass(frozen=True)
class PartitionSpec:
    idx: int
    name: str
    train_dir: Path
    test_dir: Path
    csv_path: Path
    description: str

    @property
    def display_name(self) -> str:
        return f"Partition {self.idx} - {self.description}"

    @property
    def partition_label(self) -> str:
        return f"P{self.idx}: {self.description}"


@dataclass
class PartitionResult:
    spec: PartitionSpec
    metrics: Dict[str, float]
    cm: np.ndarray
    fpr: np.ndarray
    tpr: np.ndarray
    split_stats: List[Dict]
    y_true: np.ndarray
    y_pred: np.ndarray
    y_scores: np.ndarray


@dataclass
class LossCurves:
    epochs: List[int]
    train_loss: List[float]
    val_loss: List[float]
    val_acc: List[float]
    best_epoch: int
    best_acc: float
    eval_epoch: int
    eval_acc: float


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dt", type=str)
    parser.add_argument(
        "--use-best",
        action="store_true",
        dest="use_best",
        help="Evaluate using the best checkpoint instead of the last one.",
    )
    return parser.parse_args()


def pct(value: int, total: int) -> float:
    return (100.0 * value / total) if total else 0.0


def summarize_split_stats(csv_path: Path) -> List[Dict]:
    df = pd.read_csv(csv_path)
    stats = []
    
    # Overall total
    total_subj = len(df)
    
    splits_to_process = ["train", "val", "test"]
    
    # Calculate totals across all splits
    total_sarcopenia = int((df["sarcopenia-normal"] == "sarcopenia").sum())
    total_normal = int((df["sarcopenia-normal"] == "normal").sum())
    total_stable = int(df.get("stable", pd.Series(dtype=int)).sum())
    total_unstable = int(df.get("unstable", pd.Series(dtype=int)).sum())
    grand_total_clips = total_stable + total_unstable
    
    # Add "total" row first
    stats.append({
        "split": "total",
        "subjects": total_subj,
        "subjects_pct": 100.0,
        "sarcopenia": total_sarcopenia,
        "sarcopenia_pct": pct(total_sarcopenia, total_subj),
        "normal": total_normal,
        "normal_pct": pct(total_normal, total_subj),
        "clips": grand_total_clips,
        "stable": total_stable,
        "stable_pct": pct(total_stable, grand_total_clips),
        "unstable": total_unstable,
        "unstable_pct": pct(total_unstable, grand_total_clips),
    })
    
    for split in splits_to_process:
        split_df = df[df["split"] == split]
        subj_count = len(split_df)
        
        # Subject counts
        sarcopenia = int((split_df["sarcopenia-normal"] == "sarcopenia").sum())
        normal = int((split_df["sarcopenia-normal"] == "normal").sum())
        
        # Clip counts
        stable = int(split_df.get("stable", pd.Series(dtype=int)).sum())
        unstable = int(split_df.get("unstable", pd.Series(dtype=int)).sum())
        total_clips = stable + unstable
        
        stats.append({
            "split": split,
            "subjects": subj_count,
            "subjects_pct": pct(subj_count, total_subj),
            "sarcopenia": sarcopenia,
            "sarcopenia_pct": pct(sarcopenia, subj_count),
            "normal": normal,
            "normal_pct": pct(normal, subj_count),
            "clips": total_clips,
            "stable": stable,
            "stable_pct": pct(stable, total_clips),
            "unstable": unstable,
            "unstable_pct": pct(unstable, total_clips),
        })
    return stats


def read_partition_description(base_dir: Path, partition_name: str) -> str:
    desc_path = base_dir / f"{partition_name}.txt"
    if desc_path.exists():
        text = desc_path.read_text(encoding="utf-8").strip()
        if text:
            return text.splitlines()[0]
    return partition_name


def build_partitions(dt: str) -> List[PartitionSpec]:
    dt_dir = WORK_DIR / dt
    if not dt_dir.exists():
        raise FileNotFoundError(f"Missing directory {dt_dir}")

    train_dirs = sorted(
        p for p in dt_dir.iterdir()
        if p.is_dir() and not p.name.endswith("_test")
    )

    partitions: List[PartitionSpec] = []
    for idx, train_dir in enumerate(train_dirs):
        csv_path = dt_dir / f"{train_dir.name}.csv"
        if not csv_path.exists():
            print(f"Skipping {train_dir.name}: missing {csv_path}")
            continue
        test_dir = dt_dir / f"{train_dir.name}_test"
        desc = read_partition_description(dt_dir, train_dir.name)
        partitions.append(
            PartitionSpec(
                idx=idx,
                name=train_dir.name,
                train_dir=train_dir,
                test_dir=test_dir,
                csv_path=csv_path,
                description=desc,
            )
        )

    if not partitions:
        raise FileNotFoundError(f"No partitions with logs found inside {dt_dir}")
    return partitions


def extract_loss_curves(
    logdir: Path, use_last_eval: bool, run_label: str = ""
) -> Optional[LossCurves]:
    """Extract training/validation curves from TensorBoard logs."""
    ea = event_accumulator.EventAccumulator(str(logdir))
    ea.Reload()

    # Get all scalar data
    loss_records = ea.Scalars("loss")
    val_loss_records = ea.Scalars("val_loss")
    val_acc_records = ea.Scalars("val_acc")
    epoch_records = ea.Scalars("epoch")

    # Build dictionaries mapping step -> value for each metric
    loss_by_step = {r.step: r.value for r in loss_records}
    val_loss_by_step = {r.step: r.value for r in val_loss_records}
    val_acc_by_step = {r.step: r.value for r in val_acc_records}
    epoch_by_step = {r.step: int(r.value) for r in epoch_records}

    # Group steps by epoch value
    steps_by_epoch = {}
    for step, epoch in epoch_by_step.items():
        if epoch not in steps_by_epoch:
            steps_by_epoch[epoch] = []
        steps_by_epoch[epoch].append(step)

    # Get metrics for each epoch (use last step for each epoch)
    epochs, y_loss, y_vloss, y_vacc = [], [], [], []
    for epoch in sorted(steps_by_epoch.keys()):
        # Use the last step for this epoch
        step = max(steps_by_epoch[epoch])
        if step in loss_by_step and step in val_loss_by_step and step in val_acc_by_step:
            epochs.append(epoch)
            y_loss.append(loss_by_step[step])
            y_vloss.append(val_loss_by_step[step])
            y_vacc.append(val_acc_by_step[step])

    if not epochs:
        label = f" {run_label}" if run_label else ""
        print(f"Warning: No valid epoch data found for run{label}")
        return None

    # Find best epoch
    best_idx = max(range(len(y_vacc)), key=lambda j: y_vacc[j])
    best_epoch = epochs[best_idx]
    best_acc = y_vacc[best_idx]
    epoch_to_val_acc = dict(zip(epochs, y_vacc))
    eval_epoch = epochs[-1] if use_last_eval else best_epoch
    eval_acc = epoch_to_val_acc[eval_epoch]

    return LossCurves(
        epochs=epochs,
        train_loss=y_loss,
        val_loss=y_vloss,
        val_acc=y_vacc,
        best_epoch=best_epoch,
        best_acc=best_acc,
        eval_epoch=eval_epoch,
        eval_acc=eval_acc,
    )


def plot_loss(run_i: int, curves: LossCurves, use_last_eval: bool):
    """Plot training loss, validation loss, and validation accuracy per epoch."""
    epochs = curves.epochs
    y_loss = curves.train_loss
    y_vloss = curves.val_loss
    y_vacc = curves.val_acc
    best_epoch = curves.best_epoch
    best_acc = curves.best_acc
    eval_epoch = curves.eval_epoch
    eval_acc = curves.eval_acc

    # Plot
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axs[0].plot(epochs, y_loss, label="loss")
    axs[0].plot(epochs, y_vloss, label="val_loss")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(epochs, y_vacc, label="val_acc")
    axs[1].scatter([best_epoch], [best_acc], color="red", s=40, label="best")

    eval_label = "used for test (last)" if use_last_eval else "used for test (best)"
    if eval_epoch == best_epoch:
        # Draw hollow marker on top so overlapping points remain distinguishable
        axs[1].scatter(
            [eval_epoch],
            [eval_acc],
            facecolors="none",
            edgecolors="black",
            s=80,
            linewidths=1.5,
            label=eval_label,
        )
    else:
        axs[1].scatter(
            [eval_epoch],
            [eval_acc],
            color="tab:green",
            marker="X",
            s=60,
            label=eval_label,
        )
    axs[1].set_ylabel("Val Acc")
    axs[1].set_xlabel("Epoch")
    axs[1].legend()
    axs[1].grid(True)

    fig.suptitle(f"Run {run_i} — Best val_acc at epoch {best_epoch} ({best_acc:.4f})")
    fig.tight_layout()
    return fig


def plot_confusion_matrix_and_roc(partition: PartitionSpec, csv_path: Path, output_dir: Path):
    """Plot confusion matrix and ROC curve for a single run."""
    df = pd.read_csv(csv_path)
    df["positive_confidence"] = (
        df["pred"] * df["chosen_confidence"]
        + (1 - df["pred"]) * (1 - df["chosen_confidence"])
    )

    y_true = df["gt"].to_numpy()
    y_pred = df["pred"].to_numpy()
    y_scores = df["positive_confidence"].to_numpy()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_scores)

    print(f"\nFile: {csv_path}")
    print(
        f"Accuracy: {acc:.4f} | Precision: {prec:.4f} "
        f"| Recall: {rec:.4f} | ROC-AUC: {auc:.4f}"
    )

    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_scores)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(cm, cmap="Blues")
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(["stable (0)", "unstable (1)"])
    axes[0].set_yticklabels(["stable (0)", "unstable (1)"])
    axes[0].set_xlabel("Prediction")
    axes[0].set_ylabel("Ground Truth")
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, cm[i, j], ha="center", va="center", color="black")

    axes[1].plot(fpr, tpr, label=f"AUC={auc:.3f}")
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.5)
    axes[1].set_title("ROC Curve")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    text = (
        f"Accuracy:  {acc:.3f}\n"
        f"Precision: {prec:.3f}\n"
        f"Recall:    {rec:.3f}\n"
        f"ROC-AUC:   {auc:.3f}"
    )
    axes[1].text(
        0.65,
        0.25,
        text,
        transform=axes[1].transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

    fig.tight_layout()
    out_pdf = output_dir / f"run{partition.idx}_cm.pdf"
    fig.savefig(out_pdf)
    print(f"Saving to {out_pdf}")

    out_jpg = output_dir / f"run{partition.idx}_cm.jpg"
    fig.savefig(out_jpg)
    print(f"Saving to {out_jpg}")
    plt.close(fig)

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "roc_auc": auc,
    }
    return cm, fpr, tpr, metrics, y_true, y_pred, y_scores


def evaluate_partition(
    partition: PartitionSpec,
    use_last: bool,
    output_dir: Path,
) -> Optional[PartitionResult]:
    split_stats = summarize_split_stats(partition.csv_path)
    results_file = (partition.test_dir if use_last else partition.train_dir) / "test_result.txt"
    if not results_file.exists():
        print(f"Skipping {partition.display_name}: missing {results_file}")
        return None

    cm, fpr, tpr, metrics, y_true, y_pred, y_scores = plot_confusion_matrix_and_roc(
        partition, results_file, output_dir
    )
    return PartitionResult(
        spec=partition,
        metrics=metrics,
        cm=cm,
        fpr=fpr,
        tpr=tpr,
        split_stats=split_stats,
        y_true=y_true,
        y_pred=y_pred,
        y_scores=y_scores,
    )


def plot_average_results(results: List[PartitionResult], output_dir: Path):
    if not results:
        return

    avg_cm = np.mean([res.cm for res in results], axis=0)

    accuracies = np.array([res.metrics["accuracy"] for res in results], dtype=float)
    precisions = np.array([res.metrics["precision"] for res in results], dtype=float)
    recalls = np.array([res.metrics["recall"] for res in results], dtype=float)
    aucs = np.array([res.metrics["roc_auc"] for res in results], dtype=float)

    def mean_std(values: np.ndarray) -> (float, float):
        if values.size == 0:
            return 0.0, 0.0
        if values.size == 1:
            return float(values[0]), 0.0
        return float(values.mean()), float(values.std(ddof=1))

    avg_metrics = {}
    std_metrics = {}
    avg_metrics["accuracy"], std_metrics["accuracy"] = mean_std(accuracies)
    avg_metrics["precision"], std_metrics["precision"] = mean_std(precisions)
    avg_metrics["recall"], std_metrics["recall"] = mean_std(recalls)
    avg_metrics["roc_auc"], std_metrics["roc_auc"] = mean_std(aucs)

    xs = np.linspace(0, 1, 200)
    tpr_samples = np.array(
        [np.interp(xs, res.fpr, res.tpr) for res in results],
        dtype=float,
    )
    avg_tpr = tpr_samples.mean(axis=0)
    if tpr_samples.shape[0] > 1:
        std_tpr = tpr_samples.std(axis=0, ddof=1)
    else:
        std_tpr = np.zeros_like(avg_tpr)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(avg_cm, cmap="Blues")
    axes[0].set_title("Averaged Confusion Matrix")
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(["stable (0)", "unstable (1)"])
    axes[0].set_yticklabels(["stable (0)", "unstable (1)"])
    axes[0].set_xlabel("Prediction")
    axes[0].set_ylabel("Ground Truth")
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, f"{avg_cm[i, j]:.1f}", ha="center", va="center")

    axes[1].plot(xs, avg_tpr, label="Avg ROC")
    # Shade ±1 standard deviation band around the mean ROC, if available
    upper = np.clip(avg_tpr + std_tpr, 0.0, 1.0)
    lower = np.clip(avg_tpr - std_tpr, 0.0, 1.0)
    if not np.allclose(upper, lower):
        axes[1].fill_between(
            xs,
            lower,
            upper,
            color="tab:blue",
            alpha=0.2,
            label="±1 SD",
        )
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.5)
    axes[1].set_title("Averaged ROC Curve")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    text = (
        f"Accuracy:  {avg_metrics['accuracy']:.3f} ± {std_metrics['accuracy']:.3f}\n"
        f"Precision: {avg_metrics['precision']:.3f} ± {std_metrics['precision']:.3f}\n"
        f"Recall:    {avg_metrics['recall']:.3f} ± {std_metrics['recall']:.3f}\n"
        f"ROC-AUC:   {avg_metrics['roc_auc']:.3f} ± {std_metrics['roc_auc']:.3f}"
    )
    axes[1].text(
        0.65,
        0.25,
        text,
        transform=axes[1].transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

    fig.tight_layout()
    out_pdf = output_dir / "avg_cm.pdf"
    print(f"Saving to {out_pdf}")
    fig.savefig(out_pdf)
    out_jpg = output_dir / "avg_cm.jpg"
    print(f"Saving to {out_jpg}")
    fig.savefig(out_jpg)
    plt.close(fig)


def plot_aggregate_results(results: List[PartitionResult], output_dir: Path):
    if not results:
        return None

    agg_cm = np.sum([res.cm for res in results], axis=0).astype(int)
    all_true = np.concatenate([res.y_true for res in results])
    all_pred = np.concatenate([res.y_pred for res in results])
    all_scores = np.concatenate([res.y_scores for res in results])

    total = agg_cm.sum()
    tp = agg_cm[1, 1]
    tn = agg_cm[0, 0]
    fp = agg_cm[0, 1]
    fn = agg_cm[1, 0]

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    if len(np.unique(all_true)) > 1:
        fpr, tpr, _ = roc_curve(all_true, all_scores)
        auc = roc_auc_score(all_true, all_scores)
    else:
        fpr = np.array([0.0, 1.0])
        tpr = np.array([0.0, 1.0])
        auc = float("nan")

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "roc_auc": auc,
    }

    fig, axes = plt.subplots(1, 2, figsize=(7, 5))

    axes[0].imshow(agg_cm, cmap="Blues")
    axes[0].set_title("Aggregate Confusion Matrix")
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(["stable (0)", "unstable (1)"])
    axes[0].set_yticklabels(["stable (0)", "unstable (1)"])
    axes[0].set_xlabel("Prediction")
    axes[0].set_ylabel("Ground Truth")
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, f"{agg_cm[i, j]:d}", ha="center", va="center")

    axes[1].plot(fpr, tpr, label=f"ROC-AUC={auc:.3f}" if not np.isnan(auc) else "ROC")
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.5)
    axes[1].set_title("Aggregate ROC Curve")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    text = (
        f"Accuracy:  {accuracy:.3f}\n"
        f"Precision: {precision:.3f}\n"
        f"Recall:    {recall:.3f}\n"
        f"ROC-AUC:   {auc:.3f}"
    )
    axes[1].text(
        0.65,
        0.25,
        text,
        transform=axes[1].transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

    fig.tight_layout()
    out_pdf = output_dir / "agg_cm.pdf"
    print(f"Saving to {out_pdf}")
    fig.savefig(out_pdf)
    out_jpg = output_dir / "agg_cm.jpg"
    print(f"Saving to {out_jpg}")
    fig.savefig(out_jpg)
    plt.close(fig)

    return {"cm": agg_cm, "metrics": metrics}


def build_partition_sections(
    results: List[PartitionResult],
    evaluated_on: str,
    output_dir: Path,
) -> str:
    if not results:
        return ""

    base = output_dir.as_posix()
    sections = []
    for idx, res in enumerate(results):
        stats_rows = []
        for s in res.split_stats:
            stats_rows.append(
                f"[{s['split']}], "
                f"[{s['subjects']} ({s['subjects_pct']:.1f}%)], "
                f"[{s['sarcopenia']} ({s['sarcopenia_pct']:.1f}%)], "
                f"[{s['normal']} ({s['normal_pct']:.1f}%)], "
                f"[{s['clips']}], "
                f"[{s['unstable']} ({s['unstable_pct']:.1f}%)], "
                f"[{s['stable']} ({s['stable_pct']:.1f}%)], "
            )
        stats_table_content = "\n    ".join(stats_rows)

        section = f"""
=== {res.spec.display_name}

{evaluated_on}

#table(
  columns: 7,
  [Split], [Total Subj], [Sarcopenia], [Normal], [Total Clips], [Unstable], [Stable],
  {stats_table_content}
)

#image("{base}/run{res.spec.idx}_cm.pdf")
#image("{base}/run{res.spec.idx}_loss.pdf")
""".strip()
        if idx < len(results) - 1:
            section += "\n#pagebreak()"
        sections.append(section)

    return "\n\n".join(sections)


def build_partition_table(results: List[PartitionResult]) -> str:
    if not results:
        return ""

    # Collect all metric values to find max for each
    all_accuracies = [res.metrics['accuracy'] for res in results]
    all_precisions = [res.metrics['precision'] for res in results]
    all_recalls = [res.metrics['recall'] for res in results]
    all_aucs = [res.metrics['roc_auc'] for res in results]

    max_acc = max(all_accuracies)
    max_prec = max(all_precisions)
    max_rec = max(all_recalls)
    max_auc = max(all_aucs)

    def fmt_val(val: float, max_val: float) -> str:
        s = f"{val:.3f}"
        return f"*{s}*" if val == max_val else s

    rows = []
    for res in results:
        test_stats = next((s for s in res.split_stats if s["split"] == "test"), None)
        if test_stats:
            sarc_pct = test_stats["sarcopenia_pct"]
            unstable_pct = test_stats["unstable_pct"]
        else:
            sarc_pct = 0.0
            unstable_pct = 0.0

        metrics = res.metrics
        rows.append(
            f"[{res.spec.partition_label}], "
            f"[{fmt_val(metrics['accuracy'], max_acc)}], "
            f"[{fmt_val(metrics['precision'], max_prec)}], "
            f"[{fmt_val(metrics['recall'], max_rec)}], "
            f"[{fmt_val(metrics['roc_auc'], max_auc)}], "
            f"[{sarc_pct:.1f}%], "
            f"[{unstable_pct:.1f}%],"
        )

    rows_text = "\n  ".join(rows)
    return f"""
== Partition comparison (Test Set)

#table(
  columns: 7,
  [Partition], [Accuracy], [Precision], [Recall], [ROC-AUC], [Test Sarc%], [Test Unstable%],
  {rows_text}
)
""".strip()


def build_aggregate_section(
    aggregate_info: Optional[Dict],
    results: List[PartitionResult],
    output_dir: Path,
) -> str:
    if not aggregate_info:
        return ""

    metrics = aggregate_info["metrics"]

    # Also compute mean ± SD across folds for comparison
    if results:
        accuracies = np.array(
            [res.metrics["accuracy"] for res in results],
            dtype=float,
        )
        precisions = np.array(
            [res.metrics["precision"] for res in results],
            dtype=float,
        )
        recalls = np.array(
            [res.metrics["recall"] for res in results],
            dtype=float,
        )
        aucs = np.array(
            [res.metrics["roc_auc"] for res in results],
            dtype=float,
        )

        def mean_std(values: np.ndarray) -> (float, float):
            if values.size == 0:
                return 0.0, 0.0
            if values.size == 1:
                return float(values[0]), 0.0
            return float(values.mean()), float(values.std(ddof=1))

        mean_acc, std_acc = mean_std(accuracies)
        mean_prec, std_prec = mean_std(precisions)
        mean_rec, std_rec = mean_std(recalls)
        mean_auc, std_auc = mean_std(aucs)
    else:
        mean_acc = std_acc = mean_prec = std_prec = 0.0
        mean_rec = std_rec = mean_auc = std_auc = 0.0

    base = output_dir.as_posix()
    rows = "\n  ".join(
        [
            f"[Accuracy], [{metrics['accuracy']:.3f}], "
            f"[{mean_acc:.3f} ± {std_acc:.3f}],",
            f"[Precision], [{metrics['precision']:.3f}], "
            f"[{mean_prec:.3f} ± {std_prec:.3f}],",
            f"[Recall], [{metrics['recall']:.3f}], "
            f"[{mean_rec:.3f} ± {std_rec:.3f}],",
            f"[ROC-AUC], [{metrics['roc_auc']:.3f}], "
            f"[{mean_auc:.3f} ± {std_auc:.3f}],",
        ]
    )

    return f"""
== Aggregate across folds

Metrics below pool every prediction collected from all partitions to produce a single
confusion matrix and ROC curve, highlighting the overall cross-validation performance.

#table(
  columns: 3,
  [Metric], [Aggregate], [Mean ± SD across folds],
  {rows}
)

#image("{base}/agg_cm.pdf")
""".strip()


def build_latex_cv_table(results: List[PartitionResult]) -> str:
    if not results:
        return ""

    ordered = sorted(results, key=lambda r: r.spec.idx)
    accs = [res.metrics["accuracy"] for res in ordered]
    aucs = [res.metrics["roc_auc"] for res in ordered]
    rows = []
    for res in ordered:
        acc = res.metrics["accuracy"]
        auc = res.metrics["roc_auc"]
        test_samples = len(res.y_true)
        rows.append(
            f"Fold {res.spec.idx} & {acc:.3f} & {auc:.3f} & {test_samples} \\\\"
        )

    def mean_std(values: List[float]) -> (float, float):
        if not values:
            return 0.0, 0.0
        if len(values) == 1:
            return float(values[0]), 0.0
        return float(np.mean(values)), float(np.std(values, ddof=1))

    mean_acc, std_acc = mean_std(accs)
    mean_auc, std_auc = mean_std(aucs)

    table_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Clip-level classification performance across 3-fold cross-validation.}",
        r"\label{tab:cv_results}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Fold} & \textbf{Accuracy} & \textbf{ROC-AUC} & \textbf{Test Samples} \\",
        r"\midrule",
        *rows,
        r"\midrule",
        f"Mean ± SD & {mean_acc:.3f} ± {std_acc:.3f} & {mean_auc:.3f} ± {std_auc:.3f} & -- \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(table_lines)


def plot_combined_figure(
    results: List[PartitionResult],
    loss_curves_by_idx: Dict[int, LossCurves],
    output_path: Path,
) -> None:
    if not results:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(7, 5))

    ordered = sorted(results, key=lambda r: r.spec.idx)
    # Confusion matrices (top row) and loss curves (bottom row)
    for col in range(3):
        res = ordered[col] if col < len(ordered) else None

        # Confusion matrix
        ax_cm = axes[0, col]
        if res:
            cm = res.cm
            acc = res.metrics["accuracy"]
            auc = res.metrics["roc_auc"]
            ax_cm.imshow(cm, cmap="Blues")
            ax_cm.set_title(f"Fold {res.spec.idx} (Acc {acc:.3f}, AUC {auc:.3f})")
            ax_cm.set_xticks([0, 1])
            ax_cm.set_yticks([0, 1])
            ax_cm.set_xticklabels(["stable (0)", "unstable (1)"])
            ax_cm.set_yticklabels(["stable (0)", "unstable (1)"])
            ax_cm.set_xlabel("Prediction")
            ax_cm.set_ylabel("Ground Truth")
            for i in range(2):
                for j in range(2):
                    ax_cm.text(j, i, cm[i, j], ha="center", va="center", color="black")
        else:
            ax_cm.axis("off")

        # Loss curves
        ax_loss = axes[1, col]
        if res and res.spec.idx in loss_curves_by_idx:
            curves = loss_curves_by_idx[res.spec.idx]
            ax_loss.plot(curves.epochs, curves.train_loss, label="loss")
            ax_loss.plot(curves.epochs, curves.val_loss, label="val_loss")
            ax_loss.set_title(f"Loss curves — Fold {res.spec.idx}")
            ax_loss.set_xlabel("Epoch")
            ax_loss.set_ylabel("Loss")
            ax_loss.legend()
            ax_loss.grid(True, alpha=0.3)
        else:
            ax_loss.axis("off")

    fig.tight_layout()
    fig.savefig(output_path)
    print(f"Saved combined figure to {output_path}")
    plt.close(fig)


def save_metrics_csv(results: List[PartitionResult], output_dir: Path):
    if not results:
        return

    rows = []
    for res in results:
        # Flatten split stats
        stats_dict = {}
        for s in res.split_stats:
            split = s["split"]
            stats_dict[f"{split}_subj"] = s["subjects"]
            stats_dict[f"{split}_sarc_pct"] = s["sarcopenia_pct"]
            stats_dict[f"{split}_unstable_pct"] = s["unstable_pct"]

        rows.append(
            {
                "partition": res.spec.partition_label,
                **res.metrics,
                **stats_dict
            }
        )

    metrics_path = output_dir / "partition_metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    print(f"Saved partition metrics to {metrics_path}")


def load_training_config(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def build_typst_source(
    dt: str,
    training_config: str,
    partition_sections: str,
    comparison_section: str,
    aggregate_section: str,
) -> str:
    return f"""
= Partitioning results

*Using {dt}.*

The partitioning was done using the "Double-Stratified" strategy.

== Training results

Accuracy, precision, recall and ROC-AUC (for unstable) are included in each ROC curve plot on the right.

{partition_sections}

{comparison_section}

{aggregate_section}

== Training config

All partitions use the same configuration, except for the used dataset.

```yaml
{training_config}
```

""".strip()


def compile_typst_report(typst_source: str, output_path: Path):
    proc = subprocess.run(
        ["typst", "c", "-", str(output_path / "report.pdf"), "--open"],
        input=typst_source.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    print("Compiled report. stderr:", proc.stderr.decode())


def main():
    args = parse_args()
    dt = args.dt
    partitions = build_partitions(dt)
    output_dir = REPORT_ROOT / dt
    output_dir.mkdir(parents=True, exist_ok=True)
    loss_curves_by_idx: Dict[int, LossCurves] = {}
    use_last = not args.use_best

    results: List[PartitionResult] = []
    for partition in partitions:
        logdir = partition.train_dir / LOG_SUBDIR
        if logdir.exists():
            curves = extract_loss_curves(
                logdir, use_last, run_label=str(partition.idx)
            )
            if curves:
                loss_curves_by_idx[partition.idx] = curves
                fig = plot_loss(partition.idx, curves, use_last)
                if fig:
                    out_pdf = output_dir / f"run{partition.idx}_loss.pdf"
                    print(f"Saving to {out_pdf}")
                    fig.savefig(out_pdf)

                    out_jpg = output_dir / f"run{partition.idx}_loss.jpg"
                    print(f"Saving to {out_jpg}")
                    fig.savefig(out_jpg)
                    plt.close(fig)
            else:
                print(f"Skipping loss plot for {partition.display_name}: no scalar data")
        else:
            print(f"Skipping loss plot for {partition.display_name}: missing {logdir}")

        result = evaluate_partition(partition, use_last, output_dir)
        if result:
            results.append(result)

    if not results:
        print("No evaluation data available.")
        return

    plot_average_results(results, output_dir)
    aggregate_info = plot_aggregate_results(results, output_dir)
    save_metrics_csv(results, output_dir)

    latex_table = build_latex_cv_table(results)
    if latex_table:
        print("\nLaTeX CV results table:\n")
        print(latex_table)
        print()

    combined_fig_path = PLOTS_ROOT / "cv_combined.pdf"
    plot_combined_figure(results, loss_curves_by_idx, combined_fig_path)

    evaluated_on = (
        "Evaluated on *last* epoch." if use_last else "Evaluated on *best* epoch."
    )
    partition_sections = build_partition_sections(results, evaluated_on, output_dir)
    comparison_section = build_partition_table(results)
    aggregate_section = build_aggregate_section(aggregate_info, results, output_dir)
    training_config = load_training_config(WORK_DIR / dt / "base_config.yaml")
    typst_source = build_typst_source(
        dt,
        training_config,
        partition_sections,
        comparison_section,
        aggregate_section,
    )

    compile_typst_report(typst_source, output_dir)


if __name__ == "__main__":
    main()
