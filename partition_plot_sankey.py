import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

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

# Create nodes: each partition has 3 nodes (train, val, test)
node_labels = []
node_colors = []
split_colors = {'train': 'rgba(255, 99, 71, 0.8)', 
                'val': 'rgba(50, 205, 50, 0.8)', 
                'test': 'rgba(65, 105, 225, 0.8)'}

for i in range(len(partition_paths)):
    for split in ['train', 'val', 'test']:
        node_labels.append(f"Partition {i+1}\n{split.capitalize()}")
        node_colors.append(split_colors[split])

# Create mapping: node index = partition_idx * 3 + split_idx
def get_node_idx(partition_idx, split):
    split_map = {'train': 0, 'val': 1, 'test': 2}
    return partition_idx * 3 + split_map[split]

# Build links between consecutive partitions
sources = []
targets = []
values = []
link_colors = []

for partition_idx in range(len(partition_paths) - 1):
    df_current = partition_dfs[partition_idx]
    df_next = partition_dfs[partition_idx + 1]
    
    # For each subject, track where it goes from current partition to next partition
    for _, row_curr in df_current.iterrows():
        subject = row_curr['subject']
        split_curr = row_curr['split']
        
        # Find where this subject is in the next partition
        row_next = df_next[df_next['subject'] == subject]
        if len(row_next) > 0:
            split_next = row_next.iloc[0]['split']
            
            source_idx = get_node_idx(partition_idx, split_curr)
            target_idx = get_node_idx(partition_idx + 1, split_next)
            
            sources.append(source_idx)
            targets.append(target_idx)
            values.append(1)  # Each subject counts as 1
            
            # Color the link based on source split
            link_colors.append(split_colors[split_curr].replace('0.8', '0.3'))

# Create Sankey diagram
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=node_labels,
        color=node_colors,
    ),
    link=dict(
        source=sources,
        target=targets,
        value=values,
        color=link_colors,
    )
)])

fig.update_layout(
    title="Subject Flow Across Partitions<br><sub>Each line represents one subject moving between train/val/test splits</sub>",
    font=dict(size=12, family="Arial"),
    height=800,
    width=1400,
)

fig.show()

# Print some statistics about subject movement
print("\nSubject Movement Statistics:")
print("=" * 50)
for partition_idx in range(len(partition_paths) - 1):
    df_current = partition_dfs[partition_idx]
    df_next = partition_dfs[partition_idx + 1]
    
    print(f"\nPartition {partition_idx + 1} → Partition {partition_idx + 2}:")
    
    for split_from in ['train', 'val', 'test']:
        subjects_from = set(df_current[df_current['split'] == split_from]['subject'])
        
        for split_to in ['train', 'val', 'test']:
            subjects_to = set(df_next[df_next['split'] == split_to]['subject'])
            moved = len(subjects_from & subjects_to)
            
            if moved > 0:
                print(f"  {split_from:5} → {split_to:5}: {moved:2d} subjects")
