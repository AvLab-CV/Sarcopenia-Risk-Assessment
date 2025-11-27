from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_score,
    recall_score,
    accuracy_score,
)

WORK_DIR = Path("/Users/aldo/Code/avlab/SkateFormer_synced/work_dir")
LOG_SUBDIR = Path("runs/train")
OUTPUT_DIR = Path("output/plots")
RUNS = ["part0", "part1", "part2"]

def plot_loss(run_i, logdir):
    ea = event_accumulator.EventAccumulator(str(logdir))
    ea.Reload()

    loss      = ea.Scalars("loss")
    val_loss  = ea.Scalars("val_loss")
    val_acc   = ea.Scalars("val_acc")
    epochs_tb = ea.Scalars("epoch")

    def to_xy(records):
        return [r.step for r in records], [r.value for r in records]

    # Epoch values
    _, epoch_vals = to_xy(epochs_tb)

    # Build per-epoch summaries using last occurrence
    per_epoch_loss = {}
    per_epoch_vloss = {}
    per_epoch_vacc = {}

    # --- Loss ---
    for r in loss:
        per_epoch_loss[int(r.step)] = r.value

    # --- Val Loss ---
    for r in val_loss:
        per_epoch_vloss[int(r.step)] = r.value

    # --- Val Acc ---
    for r in val_acc:
        per_epoch_vacc[int(r.step)] = r.value

    # Convert epoch list into aligned arrays
    epochs = [int(e) for e in epoch_vals]
    y_loss  = [per_epoch_loss.get(e, None) for e in epochs]
    y_vloss = [per_epoch_vloss.get(e, None) for e in epochs]
    y_vacc  = [per_epoch_vacc.get(e, None) for e in epochs]

    # Remove epochs with missing metrics
    clean = [(e, l, vl, va) for e, l, vl, va in zip(epochs, y_loss, y_vloss, y_vacc)
             if l is not None and vl is not None and va is not None]

    epochs, y_loss, y_vloss, y_vacc = zip(*clean)

    # Best epoch
    best_idx = max(range(len(y_vacc)), key=lambda j: y_vacc[j])
    best_epoch = epochs[best_idx]
    best_acc = y_vacc[best_idx]

    # --- Plot ---
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

if __name__ == "__main__":
    for i, run in enumerate(RUNS):
        logdir = WORK_DIR / run / LOG_SUBDIR
        fig = plot_loss(i, logdir)
        fig.savefig(OUTPUT_DIR / f"run{i}_loss.pdf")
        plt.close(fig)

    # Confusion matrices
    all_cms = []
    all_fprs = []
    all_tprs = []

    for run_i, run in enumerate(RUNS):
        csv_path = WORK_DIR / run / "test_result.txt"
        df = pd.read_csv(csv_path)
        df["positive_confidence"] = (
            df["pred"] * df["chosen_confidence"]
            + (1 - df["pred"]) * (1 - df["chosen_confidence"])
        )

        y_true = df["gt"].to_numpy()
        y_pred = df["pred"].to_numpy()
        y_scores = df["positive_confidence"].to_numpy()

        # metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_scores)
        print(f"\nFile: {csv_path}")
        print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | ROC-AUC: {auc:.4f}")

        # confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        all_cms.append(cm)

        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        all_fprs.append(fpr)
        all_tprs.append(tpr)

        # plot for this file
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # confusion matrix plot
        im = axes[0].imshow(cm, cmap="Blues")
        axes[0].set_title("Confusion Matrix")
        axes[0].set_xticks([0, 1])
        axes[0].set_yticks([0, 1])
        axes[0].set_xticklabels(["stable (0)", "unstable (1)"])
        axes[0].set_yticklabels(["stable (0)", "unstable (1)"])
        for i in range(2):
            for j in range(2):
                axes[0].text(j, i, cm[i, j], ha="center", va="center", color="black")

        # ROC curve plot
        axes[1].plot(fpr, tpr, label=f"AUC={auc:.3f}")
        axes[1].plot([0, 1], [0, 1], linestyle="--")
        axes[1].set_title("ROC Curve")
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].legend()
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

    # ---- Final averaged results ----
    avg_cm = np.mean(all_cms, axis=0)

    # average ROC: interpolate each curve to 200 points
    xs = np.linspace(0, 1, 200)
    interp_tprs = [np.interp(xs, fpr, tpr) for fpr, tpr in zip(all_fprs, all_tprs)]
    avg_tpr = np.mean(interp_tprs, axis=0)

    # final plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # averaged confusion matrix
    axes[0].imshow(avg_cm, cmap="Blues")
    axes[0].set_title("Averaged Confusion Matrix")
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(["stable (0)", "unstable (1)"])
    axes[0].set_yticklabels(["stable (0)", "unstable (1)"])
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, f"{avg_cm[i, j]:.1f}", ha="center", va="center")

    # averaged ROC
    axes[1].plot(xs, avg_tpr, label="Avg ROC")
    axes[1].plot([0, 1], [0, 1], linestyle="--")
    axes[1].set_title("Averaged ROC Curve")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend()
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
    fig.savefig(OUTPUT_DIR / "avg_cm.pdf")
    plt.close(fig)
