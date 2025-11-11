import seaborn
import numpy as np
import matplotlib.pyplot as plt
import einops
from tabulate import tabulate
from pathlib import Path

CLASS_LABELS = ["stable", "unstable"]
CLASSES = 2
RESULTS_DIR = Path("results")
FOLDS = 4

# load confusion matrices
cms = [
    np.loadtxt(
        open(RESULTS_DIR / f"fold{fold + 1}_cm.csv", "rb"), delimiter=",", skiprows=1
    )
    for fold in range(FOLDS)
]
cms, _ = einops.pack(cms, "* pred gt")
accuracies = np.zeros((FOLDS))
precisions = np.zeros((FOLDS, CLASSES))
recalls = np.zeros((FOLDS, CLASSES))
f1s = np.zeros((FOLDS, CLASSES))
supports = np.zeros((FOLDS, CLASSES))

for fold in range(FOLDS):
    for idx in range(CLASSES):
        precisions[fold,idx] = cms[fold,idx,idx] / np.sum(cms[fold,idx,:])
        recalls[fold,idx]    = cms[fold,idx,idx] / np.sum(cms[fold,:,idx])
        f1s[fold,idx]        = 2 * (precisions[fold,idx] * recalls[fold,idx]) / (precisions[fold,idx] + recalls[fold,idx])
        supports[fold]       = np.sum(cms[fold,:,idx])
    
accuracies = np.sum(np.diagonal(cms, axis1=-2, axis2=-1), axis=-1) / cms.sum(axis=(-2, -1))

def print_stats(accuracy, precisions, recalls, f1s, supports):
    table = [[CLASS_LABELS[i], precisions[i], recalls[i], f1s[i], supports[i]] for i in range(CLASSES)]
    print(f"{accuracy=}")
    print(tabulate(table, headers=["class", "precision", "recall", "f1", "support"]))

for fold in range(FOLDS):
    print(f"Fold {fold+1}/{FOLDS}")
    print_stats(accuracies[fold], precisions[fold], recalls[fold], f1s[fold], supports[fold])
    print()

print("Mean over folds:")
print_stats(
    np.mean(accuracies, axis=0),
    np.mean(precisions, axis=0),
    np.mean(recalls, axis=0),
    np.mean(f1s, axis=0),
    np.mean(supports, axis=0),
)
print()

print("Std over folds:")
print_stats(
    np.std(accuracies, axis=0, ddof=1),
    np.std(precisions, axis=0, ddof=1),
    np.std(recalls, axis=0, ddof=1),
    np.std(f1s, axis=0, ddof=1),
    np.std(supports, axis=0, ddof=1),
)
print()

# ax = seaborn.heatmap(cms.mean(axis=0), xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, annot=True, square=True, cmap='Blues')
# ax.set_xlabel('Actual')
# ax.set_ylabel('Predicted')
# plt.show()
