import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("partition_dir", type=Path)
parser.add_argument("partition_count", type=int)
args = parser.parse_args()
PARTITIONS_COUNT = args.partition_count
PARTITION_DIR = args.partition_dir
print(f"Looking for {PARTITIONS_COUNT} partitions in `{PARTITION_DIR}`")
partition_paths = [
    PARTITION_DIR / f"partition{partition}.csv"
    for partition in range(PARTITIONS_COUNT)
]


# Load all partition data
partition_dfs = [pd.read_csv(partition) for partition in partition_paths]

# Extract test subjects for each partition
test_subjects = []
val_subjects = []
for df in partition_dfs:
    test_subjects.append(set(df[df['split'] == 'test']['subject'].values))
    val_subjects.append(set(df[df['split'] == 'val']['subject'].values))

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# ============= SUBPLOT 1: Test Set Composition =============
ax1.set_title('Test set composition: Which subjects were tested before?', 
              fontsize=14, fontweight='bold', pad=20)

all_tested = set()
bar_data = {'new': [], 'reused': []}

for i, test_set in enumerate(test_subjects):
    new_subjects = test_set - all_tested
    reused_subjects = test_set & all_tested
    
    bar_data['new'].append(len(new_subjects))
    bar_data['reused'].append(len(reused_subjects))
    
    all_tested.update(test_set)

x = np.arange(len(partition_paths))
width = 0.6

bars1 = ax1.bar(x, bar_data['new'], width, label='New (never tested before)', 
                color='#2ecc71', alpha=0.8)
bars2 = ax1.bar(x, bar_data['reused'], width, bottom=bar_data['new'],
                label='Reused (tested in previous partition)', color='#e74c3c', alpha=0.8)

# Add value labels on bars
for i, (new, reused) in enumerate(zip(bar_data['new'], bar_data['reused'])):
    total = new + reused
    if new > 0:
        ax1.text(i, new/2, f'{new}', ha='center', va='center', 
                fontweight='bold', fontsize=11)
    if reused > 0:
        ax1.text(i, new + reused/2, f'{reused}', ha='center', va='center',
                fontweight='bold', fontsize=11, color='white')
    ax1.text(i, total + 0.3, f'{total}', ha='center', va='bottom',
            fontweight='bold', fontsize=10, color='black')

ax1.set_xlabel('Partition', fontsize=12)
ax1.set_ylabel('Number of Subjects', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels([f'Partition {i+1}' for i in range(len(partition_paths))])
ax1.legend(fontsize=11, loc='upper right')
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim(0, max([sum(x) for x in zip(bar_data['new'], bar_data['reused'])]) * 1.15)

# ============= SUBPLOT 2: Pairwise Overlap Matrix =============
ax2.set_title('Test set overlap between partitions (IoU)', 
              fontsize=14, fontweight='bold', pad=20)

n_partitions = len(test_subjects)
overlap_matrix = np.zeros((n_partitions, n_partitions))

for i in range(n_partitions):
    for j in range(n_partitions):
        if i != j:
            intersection = len(test_subjects[i] & test_subjects[j])
            union = len(test_subjects[i] | test_subjects[j])
            overlap_matrix[i, j] = intersection / union if union > 0 else 0

im = ax2.imshow(overlap_matrix, cmap='RdYlGn_r', vmin=0, vmax=1, aspect='auto')

# Add text annotations
for i in range(n_partitions):
    for j in range(n_partitions):
        if i != j:
            text = ax2.text(j, i, f'{overlap_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontweight='bold')
        else:
            ax2.text(j, i, '—', ha="center", va="center", 
                    color="gray", fontsize=16)

ax2.set_xticks(np.arange(n_partitions))
ax2.set_yticks(np.arange(n_partitions))
ax2.set_xticklabels([f'Partition {i+1}' for i in range(n_partitions)])
ax2.set_yticklabels([f'Partition {i+1}' for i in range(n_partitions)])
ax2.set_xlabel('Partition', fontsize=12)
ax2.set_ylabel('Partition', fontsize=12)

# Colorbar
cbar = plt.colorbar(im, ax=ax2)
cbar.set_label('Overlap (0=no overlap, 1=identical)', rotation=270, labelpad=20)

plt.tight_layout()
plt.savefig("reuse.png")

# ============= Print Statistics =============
print("\n" + "="*70)
print("TEST SET REUSE STATISTICS")
print("="*70)

all_tested_cumulative = set()
for i, test_set in enumerate(test_subjects):
    new_subjects = test_set - all_tested_cumulative
    reused_subjects = test_set & all_tested_cumulative
    
    print(f"\nPartition {i+1}:")
    print(f"  Total test subjects: {len(test_set)}")
    print(f"  New subjects (never tested): {len(new_subjects)} ({100*len(new_subjects)/len(test_set):.1f}%)")
    print(f"  Reused subjects (tested before): {len(reused_subjects)} ({100*len(reused_subjects)/len(test_set) if len(test_set) > 0 else 0:.1f}%)")
    
    if reused_subjects:
        print(f"  Reused subject IDs: {sorted(list(reused_subjects))}")
    
    all_tested_cumulative.update(test_set)

print(f"\n" + "="*70)
print(f"TOTAL unique subjects ever tested: {len(all_tested_cumulative)}")
print(f"Total subjects in dataset: {len(set().union(*[set(df['subject'].values) for df in partition_dfs]))}")

coverage = len(all_tested_cumulative) / len(set().union(*[set(df['subject'].values) for df in partition_dfs]))
print(f"Test coverage: {100*coverage:.1f}% of all subjects are tested at least once")
print("="*70)

# Calculate average pairwise overlap
upper_triangle = overlap_matrix[np.triu_indices_from(overlap_matrix, k=1)]
avg_overlap = np.mean(upper_triangle)
max_overlap = np.max(upper_triangle)

print(f"\nAverage pairwise test set overlap: {avg_overlap:.3f}")
print(f"Maximum pairwise test set overlap: {max_overlap:.3f}")

if avg_overlap > 0.3:
    print("\n⚠️  WARNING: High overlap detected! Partitions may not be diverse enough.")
    print("   Consider regenerating with lower --max_overlap parameter.")
elif avg_overlap < 0.15:
    print("\n✓ Good diversity! Test sets have minimal overlap across partitions.")
else:
    print("\n✓ Moderate diversity. Overlap is acceptable.")

plt.show()
