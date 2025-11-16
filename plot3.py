import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

FOLDS = [
    Path("folds_diverse/fold1_subjects.csv"),
    Path("folds_diverse/fold2_subjects.csv"),
    Path("folds_diverse/fold3_subjects.csv"),
    Path("folds_diverse/fold4_subjects.csv"),
]

# Load all fold data
fold_dfs = [pd.read_csv(fold) for fold in FOLDS]

# Create nodes: each fold has 3 nodes (train, val, test)
node_labels = []
node_colors = []
split_colors = {'train': 'rgba(255, 99, 71, 0.8)', 
                'val': 'rgba(50, 205, 50, 0.8)', 
                'test': 'rgba(65, 105, 225, 0.8)'}

for i in range(len(FOLDS)):
    for split in ['train', 'val', 'test']:
        node_labels.append(f"Fold {i+1}\n{split.capitalize()}")
        node_colors.append(split_colors[split])

# Create mapping: node index = fold_idx * 3 + split_idx
def get_node_idx(fold_idx, split):
    split_map = {'train': 0, 'val': 1, 'test': 2}
    return fold_idx * 3 + split_map[split]

# Build links between consecutive folds
sources = []
targets = []
values = []
link_colors = []

for fold_idx in range(len(FOLDS) - 1):
    df_current = fold_dfs[fold_idx]
    df_next = fold_dfs[fold_idx + 1]
    
    # For each subject, track where it goes from current fold to next fold
    for _, row_curr in df_current.iterrows():
        subject = row_curr['subject']
        split_curr = row_curr['split']
        
        # Find where this subject is in the next fold
        row_next = df_next[df_next['subject'] == subject]
        if len(row_next) > 0:
            split_next = row_next.iloc[0]['split']
            
            source_idx = get_node_idx(fold_idx, split_curr)
            target_idx = get_node_idx(fold_idx + 1, split_next)
            
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
    title="Subject Flow Across Folds<br><sub>Each line represents one subject moving between train/val/test splits</sub>",
    font=dict(size=12, family="Arial"),
    height=800,
    width=1400,
)

fig.show()

# Print some statistics about subject movement
print("\nSubject Movement Statistics:")
print("=" * 50)
for fold_idx in range(len(FOLDS) - 1):
    df_current = fold_dfs[fold_idx]
    df_next = fold_dfs[fold_idx + 1]
    
    print(f"\nFold {fold_idx + 1} → Fold {fold_idx + 2}:")
    
    for split_from in ['train', 'val', 'test']:
        subjects_from = set(df_current[df_current['split'] == split_from]['subject'])
        
        for split_to in ['train', 'val', 'test']:
            subjects_to = set(df_next[df_next['split'] == split_to]['subject'])
            moved = len(subjects_from & subjects_to)
            
            if moved > 0:
                print(f"  {split_from:5} → {split_to:5}: {moved:2d} subjects")
