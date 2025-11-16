import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FOLDS = [
    Path("folds/fold1_subjects.csv"),
    Path("folds/fold2_subjects.csv"),
    Path("folds/fold3_subjects.csv"),
    Path("folds/fold4_subjects.csv"),
]

# Collect all subjects and their splits per fold
all_subjects = set()
fold_data = []
for fold_path in FOLDS:
    df = pd.read_csv(fold_path)
    all_subjects.update(df['subject'].values)
    fold_data.append(df)

# Sort subjects by their first appearance order (fold 1: train, val, test)
df1 = fold_data[0]
ordered_subjects = []
for split in ['train', 'val', 'test']:
    split_subjects = df1[df1['split'] == split]['subject'].values
    ordered_subjects.extend(sorted(split_subjects))

# Create matrix: rows=subjects, cols=folds, values=split (0=train, 1=val, 2=test)
split_to_num = {'train': 0, 'val': 1, 'test': 2}
matrix = np.zeros((len(ordered_subjects), len(FOLDS)))

for fold_idx, df in enumerate(fold_data):
    for _, row in df.iterrows():
        subj_idx = ordered_subjects.index(row['subject'])
        matrix[subj_idx, fold_idx] = split_to_num[row['split']]

# Plot
fig, ax = plt.subplots(figsize=(8, 12))
im = ax.imshow(matrix, aspect='auto', cmap='RdYlBu_r', vmin=0, vmax=2)

# Colorbar
cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2])
cbar.set_ticklabels(['Train', 'Val', 'Test'])

# Labels
ax.set_xlabel('Fold', fontsize=12)
ax.set_ylabel('Subject', fontsize=12)
ax.set_xticks(range(len(FOLDS)))
ax.set_xticklabels([f'Fold {i+1}' for i in range(len(FOLDS))])
ax.set_title('Subject Assignment Across Folds', fontsize=14, pad=20)

# Add horizontal lines to separate original splits
df1_train_count = len(df1[df1['split'] == 'train'])
df1_val_count = len(df1[df1['split'] == 'val'])
ax.axhline(y=df1_train_count - 0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.axhline(y=df1_train_count + df1_val_count - 0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)

plt.tight_layout()
plt.show()
