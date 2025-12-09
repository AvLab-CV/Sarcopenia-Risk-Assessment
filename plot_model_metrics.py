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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dt", type=str)
    parser.add_argument(
        "--uselast",
        "--use-last",
        action="store_true",
        dest="use_last",
        help="Evaluate using the final checkpoint instead of the best one.",
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


def plot_loss(run_i, logdir, use_last_eval: bool):
    """Plot training loss, validation loss, and validation accuracy per epoch."""
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
        print(f"Warning: No valid epoch data found for run {run_i}")
        return None

    # Find best epoch
    best_idx = max(range(len(y_vacc)), key=lambda j: y_vacc[j])
    best_epoch = epochs[best_idx]
    best_acc = y_vacc[best_idx]
    epoch_to_val_acc = dict(zip(epochs, y_vacc))
    eval_epoch = epochs[-1] if use_last_eval else best_epoch
    eval_acc = epoch_to_val_acc[eval_epoch]

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
    return cm, fpr, tpr, metrics


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

    cm, fpr, tpr, metrics = plot_confusion_matrix_and_roc(partition, results_file, output_dir)
    return PartitionResult(
        spec=partition,
        metrics=metrics,
        cm=cm,
        fpr=fpr,
        tpr=tpr,
        split_stats=split_stats,
    )


def plot_average_results(results: List[PartitionResult], output_dir: Path):
    if not results:
        return

    avg_cm = np.mean([res.cm for res in results], axis=0)
    avg_metrics = {
        "accuracy": np.mean([res.metrics["accuracy"] for res in results]),
        "precision": np.mean([res.metrics["precision"] for res in results]),
        "recall": np.mean([res.metrics["recall"] for res in results]),
        "roc_auc": np.mean([res.metrics["roc_auc"] for res in results]),
    }

    xs = np.linspace(0, 1, 200)
    avg_tpr = np.mean(
        [np.interp(xs, res.fpr, res.tpr) for res in results],
        axis=0,
    )

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
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.5)
    axes[1].set_title("Averaged ROC Curve")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    text = (
        f"Accuracy:  {avg_metrics['accuracy']:.3f}\n"
        f"Precision: {avg_metrics['precision']:.3f}\n"
        f"Recall:    {avg_metrics['recall']:.3f}\n"
        f"ROC-AUC:   {avg_metrics['roc_auc']:.3f}"
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
) -> str:
    return f"""
= Ablation study: different data-splitting strategies

*Using {dt}.*

== Explanation of partitioning methods

- *Random Split* – Subjects are shuffled and partitioned purely by count into train/val/test, so clip or label composition may differ across splits. Use as a baseline for comparison.
- *Stratified by Group* – Sarcopenia and normal subjects are split independently with the same ratios, then recombined. This keeps each set’s sarcopenia/normal proportions aligned with the overall dataset.
- *Balanced 1:1 Ratio* – First undersamples the larger subject group so sarcopenia and normal have equal counts, then applies the group-wise stratified split. Ensures balanced subject representation but sacrifices some data.
- *Clip-Balanced* – Sorts subjects by clip count and greedily assigns each subject to the split with the largest clip deficit relative to the desired ratio. Aims to equalize total clips per split without per-group guarantees.
- *Stratified by Stability* – Groups subjects by their stability profile (all stable, all unstable, mixed) and splits each category separately before recombining. Keeps these behavioral patterns evenly distributed.
- *Double Stratified* – Adds another layer to the stability stratification by also separating sarcopenia vs. normal. Each group-pattern combination is split individually, yielding the most controlled, fine-grained balance across both clinical label and stability profile.


== Training results

Accuracy, precision, recall and ROC-AUC (for unstable) are included in each ROC curve plot on the right.

{partition_sections}

{comparison_section}

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

    results: List[PartitionResult] = []
    for partition in partitions:
        logdir = partition.train_dir / LOG_SUBDIR
        if logdir.exists():
            fig = plot_loss(partition.idx, logdir, args.use_last)
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

        result = evaluate_partition(partition, args.use_last, output_dir)
        if result:
            results.append(result)

    if not results:
        print("No evaluation data available.")
        return

    plot_average_results(results, output_dir)
    save_metrics_csv(results, output_dir)

    evaluated_on = (
        "Evaluated on *last* epoch." if args.use_last else "Evaluated on *best* epoch."
    )
    partition_sections = build_partition_sections(results, evaluated_on, output_dir)
    comparison_section = build_partition_table(results)
    training_config = load_training_config(WORK_DIR / dt / "base_config.yaml")
    typst_source = build_typst_source(
        dt,
        training_config,
        partition_sections,
        comparison_section,
    )

    compile_typst_report(typst_source, output_dir)


if __name__ == "__main__":
    main()
