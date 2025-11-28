import argparse
import subprocess
from pathlib import Path

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

parser = argparse.ArgumentParser()
parser.add_argument("dt", type=str)
parser.add_argument("--uselast", type=str, default=False)
args = parser.parse_args()

DT = args.dt
# DT = "20251128_1848_partition_skel"
WORK_DIR = Path("/Users/aldo/Code/avlab/SkateFormer_synced/work_dir")
LOG_SUBDIR = Path("runs/train")
OUTPUT_DIR = Path(f"output/reports/{DT}/")
RUNS = [
    f"{DT}_part0",
    f"{DT}_part1",
    f"{DT}_part2"
]


def plot_loss(run_i, logdir):
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

    # Plot
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axs[0].plot(epochs, y_loss, label="loss")
    axs[0].plot(epochs, y_vloss, label="val_loss")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(epochs, y_vacc, label="val_acc")
    axs[1].scatter([best_epoch], [best_acc], color="red", s=40, label="best")
    axs[1].set_ylabel("Val Acc")
    axs[1].set_xlabel("Epoch")
    axs[1].legend()
    axs[1].grid(True)

    fig.suptitle(f"Run {run_i} — Best val_acc at epoch {best_epoch} ({best_acc:.4f})")
    fig.tight_layout()
    return fig


def plot_confusion_matrix_and_roc(run_i, csv_path):
    """Plot confusion matrix and ROC curve for a single run."""
    df = pd.read_csv(csv_path)
    df["positive_confidence"] = (
        df["pred"] * df["chosen_confidence"]
        + (1 - df["pred"]) * (1 - df["chosen_confidence"])
    )

    y_true = df["gt"].to_numpy()
    y_pred = df["pred"].to_numpy()
    y_scores = df["positive_confidence"].to_numpy()

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_scores)
    
    print(f"\nFile: {csv_path}")
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | ROC-AUC: {auc:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_scores)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Confusion matrix
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

    # ROC curve
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
        0.65, 0.25, text,
        transform=axes[1].transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
    )

    fig.tight_layout()
    out = OUTPUT_DIR / f"run{run_i}_cm.pdf"
    print(f"Saving to {out}")
    fig.savefig(out)
    plt.close(fig)

    return cm, fpr, tpr, acc, prec, rec, auc


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    all_cms = []
    all_fprs = []
    all_tprs = []
    all_metrics = []

    # Single loop for all runs
    for run_i, run in enumerate(RUNS):
        # Plot loss curves
        logdir = WORK_DIR / run / LOG_SUBDIR
        fig = plot_loss(run_i, logdir)
        if fig:
            out = OUTPUT_DIR / f"run{run_i}_loss.pdf"
            print(f"Saving to {out}")
            fig.savefig(out)
            plt.close(fig)

        # Plot confusion matrix and ROC
        if args.uselast:
            csv_path = WORK_DIR / (run + "_test") / "test_result.txt"
        else:
            csv_path = WORK_DIR / run / "test_result.txt"
            
        if csv_path.exists():
            cm, fpr, tpr, acc, prec, rec, auc = plot_confusion_matrix_and_roc(run_i, csv_path)
            all_cms.append(cm)
            all_fprs.append(fpr)
            all_tprs.append(tpr)
            all_metrics.append({"acc": acc, "prec": prec, "rec": rec, "auc": auc})

    # Plot averaged results
    if all_cms:
        avg_cm = np.mean(all_cms, axis=0)
        avg_metrics = {
            k: np.mean([m[k] for m in all_metrics]) 
            for k in ["acc", "prec", "rec", "auc"]
        }

        # Average ROC: interpolate each curve to 200 points
        xs = np.linspace(0, 1, 200)
        interp_tprs = [np.interp(xs, fpr, tpr) for fpr, tpr in zip(all_fprs, all_tprs)]
        avg_tpr = np.mean(interp_tprs, axis=0)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Averaged confusion matrix
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

        # Averaged ROC
        axes[1].plot(xs, avg_tpr, label="Avg ROC")
        axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.5)
        axes[1].set_title("Averaged ROC Curve")
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        text = (
            f"Accuracy:  {avg_metrics['acc']:.3f}\n"
            f"Precision: {avg_metrics['prec']:.3f}\n"
            f"Recall:    {avg_metrics['rec']:.3f}\n"
            f"ROC-AUC:   {avg_metrics['auc']:.3f}"
        )
        axes[1].text(
            0.65, 0.25, text,
            transform=axes[1].transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
        )

        fig.tight_layout()
        out = OUTPUT_DIR / "avg_cm.pdf"
        print(f"Saving to {out}")
        fig.savefig(out)
        plt.close(fig)

    # Report
    # Get training yaml from run 1 (in theory all runs should have the same)
    with open(WORK_DIR / RUNS[0] / "config.yaml", 'r') as file:
        training_config = file.read()
    evaluated_on = "Evaluated on *last* epoch." if args.uselast else "Evaluated on *best* epoch."
    typst_source = f"""
= Latest results for training SkateFormer

*Using {DT}.*

Training configuration from partition 0
(in theory all partitions should use the same configuration, except for the dataloader path.)

```yaml
{training_config}
```

== Training results 

Accuracy, precision, recall and ROC-AUC (for unstable) are included in each ROC curve plot on the right.

=== Partition 0 - Balanced 1:1 ratio

{evaluated_on}

#image("{OUTPUT_DIR}/run0_cm.pdf")
#image("{OUTPUT_DIR}/run0_loss.pdf")

#pagebreak()

=== Partition 1 - Double stratified

{evaluated_on}

#image("{OUTPUT_DIR}/run1_cm.pdf")
#image("{OUTPUT_DIR}/run1_loss.pdf")
#pagebreak()

=== Partition 2 - Random split (baseline)

{evaluated_on}

#image("{OUTPUT_DIR}/run2_cm.pdf")
#image("{OUTPUT_DIR}/run2_loss.pdf")
#pagebreak()

// === Average result
// #image("{OUTPUT_DIR}/avg_cm.pdf")
    """

    proc = subprocess.run(
        ["typst", "c", "-", str(OUTPUT_DIR / "report.pdf"), "--open"],
        input=typst_source.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    print("Compiled report. stderr:", proc.stderr.decode())
