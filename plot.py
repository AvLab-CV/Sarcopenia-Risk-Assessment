import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

folds = [
    [1, 4, 6, 7, 8, 14, 19, 20, 21, 25, 27, 29, 48, 60, 63, 75, 88, 90, 103, 108, 112, 118, 123, 129, 132, 136, 138],
    [16, 18, 25, 26, 35, 49, 51, 54, 60, 65, 73, 78, 92, 101, 102, 112, 116, 117, 119, 128, 135, 139],
    [4, 7, 16, 21, 26, 30, 35, 40, 41, 50, 57, 62, 65, 68, 71, 82, 84, 85, 95, 101, 112, 117, 124, 125, 130, 131, 134, 139],
    [0, 24, 37, 40, 43, 44, 45, 49, 52, 59, 61, 66, 67, 69, 70, 74, 77, 90, 92, 94, 100, 101, 108, 116, 118, 121, 137, 138]
]

# Change subject IDX to match order of appearance. This just makes the plot cleaner
# and easier to understand.
counter = 0
old_to_new_idx = {}
new_folds = []
for fold in folds:
    new_fold = []
    for x in fold:
        if x not in old_to_new_idx:
            old_to_new_idx[x] = counter
            counter += 1
        new_fold.append(old_to_new_idx[x])
    new_folds.append(new_fold)
folds = new_folds

# Plot
colors = cm.rainbow(np.linspace(0, 1, len(folds)))
for fold_idx, (fold, c) in enumerate(zip(folds, colors)):
    x = np.array(fold)
    y = np.full_like(x, fold_idx)
    plt.scatter(x, y, color=c)
plt.show()
