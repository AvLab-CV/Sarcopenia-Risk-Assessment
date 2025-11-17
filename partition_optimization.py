#!/usr/bin/env python3
"""
Multi-objective partition optimization for subject-level dataset splitting.

This script partitions subjects into P partitions, each with train/val/test sets,
optimizing for:
1. Clip count distribution (p_train, p_val, p_test)
2. Stable/unstable clip distribution preservation
3. No subject overlap between test sets (most important)
"""

import pandas as pd
import numpy as np
import argparse
import os
from typing import Dict
import random


def calculate_targets(df: pd.DataFrame, p_train: float, p_val: float, p_test: float) -> Dict:
    """Calculate target clip counts and stable/unstable ratios."""
    total_clips = df['clip_count'].sum()
    total_stable = df['stable'].sum()
    total_unstable = df['unstable'].sum()
    total_all = total_stable + total_unstable
    
    unstable_ratio = total_unstable / total_all if total_all > 0 else 0
    
    return {
        'total_clips': total_clips,
        'total_stable': total_stable,
        'total_unstable': total_unstable,
        'unstable_ratio': unstable_ratio,
        'target_train_clips': p_train * total_clips,
        'target_val_clips': p_val * total_clips,
        'target_test_clips': p_test * total_clips,
    }


def select_best_subject_for_ratio(
    candidates: list,
    current_stable: int,
    current_unstable: int,
    target_ratio: float,
    excluded: set,
    df: pd.DataFrame
) -> tuple:
    """
    Select the subject from candidates that best maintains the target unstable ratio.
    Returns (best_idx, best_stable, best_unstable, best_clips) or (None, 0, 0, 0) if no valid candidate.
    """
    if not candidates:
        return None, 0, 0, 0
    
    best_idx = None
    best_error = float('inf')
    best_stable = 0
    best_unstable = 0
    best_clips = 0
    
    current_total = current_stable + current_unstable
    current_ratio = current_unstable / current_total if current_total > 0 else 0
    
    for idx in candidates:
        if idx in excluded:
            continue
        
        subject_stable = df.loc[idx, 'stable']
        subject_unstable = df.loc[idx, 'unstable']
        subject_clips = df.loc[idx, 'clip_count']
        
        new_stable = current_stable + subject_stable
        new_unstable = current_unstable + subject_unstable
        new_total = new_stable + new_unstable
        
        if new_total == 0:
            continue
        
        new_ratio = new_unstable / new_total
        error = abs(new_ratio - target_ratio)
        
        # Prefer subjects that bring us closer to target
        # Also prefer unstable subjects if we're below target, stable if above
        if error < best_error:
            best_error = error
            best_idx = idx
            best_stable = subject_stable
            best_unstable = subject_unstable
            best_clips = subject_clips
        elif abs(error - best_error) < 0.01:  # If similar error, prefer based on direction
            if current_ratio < target_ratio and subject_unstable > 0:
                # We need more unstable, prefer this one
                best_error = error
                best_idx = idx
                best_stable = subject_stable
                best_unstable = subject_unstable
                best_clips = subject_clips
            elif current_ratio > target_ratio and subject_stable > 0:
                # We need more stable, prefer this one
                best_error = error
                best_idx = idx
                best_stable = subject_stable
                best_unstable = subject_unstable
                best_clips = subject_clips
    
    return best_idx, best_stable, best_unstable, best_clips


def assign_subjects_to_partitions(
    df: pd.DataFrame,
    num_partitions: int,
    p_train: float,
    p_val: float,
    p_test: float,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Assign subjects to partitions and splits.
    
    Each partition is a complete split of the dataset. Subjects can appear in
    multiple partitions but only in one test set total (requirement 3).
    
    Strategy:
    1. First assign test sets (no overlap - most important)
    2. Then assign validation sets (minimize overlap)
    3. Finally assign training sets (subjects can be train in multiple partitions)
    4. Balance clip counts and stable/unstable ratios
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    targets = calculate_targets(df, p_train, p_val, p_test)
    
    # Create result dataframe - we'll have multiple rows per subject (one per partition)
    result_rows = []
    
    # Shuffle subjects for randomness
    subject_indices = df.index.tolist()
    np.random.shuffle(subject_indices)
    
    # Separate subjects by stable/unstable ratio for better distribution
    subjects_by_type = {
        'all_stable': df[df['unstable'] == 0].index.tolist(),
        'all_unstable': df[df['stable'] == 0].index.tolist(),
        'mixed': df[(df['stable'] > 0) & (df['unstable'] > 0)].index.tolist(),
    }
    
    # Shuffle each category
    for key in subjects_by_type:
        np.random.shuffle(subjects_by_type[key])
    
    # Track assignments per partition
    partition_assignments = {p: {'train': [], 'val': [], 'test': []} for p in range(num_partitions)}
    partition_clips = {p: {'train': 0, 'val': 0, 'test': 0} for p in range(num_partitions)}
    partition_stable = {p: {'train': 0, 'val': 0, 'test': 0} for p in range(num_partitions)}
    partition_unstable = {p: {'train': 0, 'val': 0, 'test': 0} for p in range(num_partitions)}
    
    # Track which subjects are already in test/val sets (to avoid overlap)
    subjects_in_test = set()
    subjects_in_val = set()
    
    # PHASE 1: Assign test sets (no overlap - most important)
    # Each partition needs target_test_clips, and no subject can be in multiple test sets
    test_target = targets['target_test_clips']
    target_unstable_ratio = targets['unstable_ratio']
    
    # Collect all candidates for test sets
    all_test_candidates = []
    for category in ['all_stable', 'all_unstable', 'mixed']:
        all_test_candidates.extend(subjects_by_type[category])
    np.random.shuffle(all_test_candidates)
    
    # Distribute test subjects across partitions, actively balancing unstable ratio
    for partition in range(num_partitions):
        current_test_clips = 0
        current_test_unstable = 0
        current_test_stable = 0
        
        while current_test_clips < test_target * 0.98:
            # Select best subject that maintains ratio
            best_idx, best_stable, best_unstable, best_clips = select_best_subject_for_ratio(
                all_test_candidates,
                current_test_stable,
                current_test_unstable,
                target_unstable_ratio,
                subjects_in_test,
                df
            )
            
            if best_idx is None:
                break  # No more candidates
            
            partition_assignments[partition]['test'].append(best_idx)
            partition_clips[partition]['test'] += best_clips
            partition_stable[partition]['test'] += best_stable
            partition_unstable[partition]['test'] += best_unstable
            subjects_in_test.add(best_idx)
            current_test_clips += best_clips
            current_test_stable += best_stable
            current_test_unstable += best_unstable
            
            if current_test_clips >= test_target:
                break
    
    # PHASE 2: Assign validation sets (minimize overlap)
    val_target = targets['target_val_clips']
    
    # Collect candidates for validation (not in test)
    val_candidates = [idx for idx in subject_indices if idx not in subjects_in_test]
    np.random.shuffle(val_candidates)
    
    for partition in range(num_partitions):
        current_val_clips = 0
        current_val_unstable = 0
        current_val_stable = 0
        
        # Track subjects already in this partition's val set to avoid overlap
        partition_val_subjects = set()
        
        while current_val_clips < val_target * 0.98:
            # Select best subject that maintains ratio and avoids overlap
            excluded = subjects_in_val | partition_val_subjects
            best_idx, best_stable, best_unstable, best_clips = select_best_subject_for_ratio(
                val_candidates,
                current_val_stable,
                current_val_unstable,
                target_unstable_ratio,
                excluded,
                df
            )
            
            if best_idx is None:
                # If no non-overlapping candidate, allow overlap but still maintain ratio
                best_idx, best_stable, best_unstable, best_clips = select_best_subject_for_ratio(
                    val_candidates,
                    current_val_stable,
                    current_val_unstable,
                    target_unstable_ratio,
                    partition_val_subjects,  # Only exclude from this partition
                    df
                )
                if best_idx is None:
                    break
            
            partition_assignments[partition]['val'].append(best_idx)
            partition_clips[partition]['val'] += best_clips
            partition_stable[partition]['val'] += best_stable
            partition_unstable[partition]['val'] += best_unstable
            subjects_in_val.add(best_idx)
            partition_val_subjects.add(best_idx)
            current_val_clips += best_clips
            current_val_stable += best_stable
            current_val_unstable += best_unstable
            
            if current_val_clips >= val_target:
                break
    
    # PHASE 3: Assign remaining subjects to training sets
    # Subjects can be in training sets of multiple partitions
    train_target = targets['target_train_clips']
    
    # All subjects not in test can potentially be in train
    train_candidates = [idx for idx in subject_indices if idx not in subjects_in_test]
    np.random.shuffle(train_candidates)
    
    for partition in range(num_partitions):
        # Get subjects already assigned to this partition (test/val)
        already_assigned = set(partition_assignments[partition]['test'] + 
                              partition_assignments[partition]['val'])
        
        current_train_clips = partition_clips[partition]['train']
        current_train_stable = partition_stable[partition]['train']
        current_train_unstable = partition_unstable[partition]['train']
        
        # Assign subjects to training set until we reach target, maintaining ratio
        while current_train_clips < train_target * 1.05:
            # Select best subject that maintains ratio
            best_idx, best_stable, best_unstable, best_clips = select_best_subject_for_ratio(
                train_candidates,
                current_train_stable,
                current_train_unstable,
                target_unstable_ratio,
                already_assigned,
                df
            )
            
            if best_idx is None:
                break  # No more candidates
            
            partition_assignments[partition]['train'].append(best_idx)
            partition_clips[partition]['train'] += best_clips
            partition_stable[partition]['train'] += best_stable
            partition_unstable[partition]['train'] += best_unstable
            current_train_clips += best_clips
            current_train_stable += best_stable
            current_train_unstable += best_unstable
            
            if current_train_clips >= train_target * 1.05:
                break
    
    # Build result dataframe
    for partition in range(num_partitions):
        for split in ['train', 'val', 'test']:
            for idx in partition_assignments[partition][split]:
                row = df.loc[idx].copy()
                row['partition'] = partition
                row['split'] = split
                result_rows.append(row)
    
    result_df = pd.DataFrame(result_rows).reset_index(drop=True)
    return result_df


def validate_partitions(result_df: pd.DataFrame, num_partitions: int) -> Dict:
    """Validate that partitions meet requirements."""
    issues = []
    warnings = []
    
    # Check test set overlaps (using 'subject' column to identify subjects)
    test_sets = {}
    for p in range(num_partitions):
        test_df = result_df[(result_df['partition'] == p) & (result_df['split'] == 'test')]
        test_sets[p] = set(test_df['subject'].values)
    
    for p1 in range(num_partitions):
        for p2 in range(p1 + 1, num_partitions):
            overlap = test_sets[p1] & test_sets[p2]
            if overlap:
                issues.append(f"Test sets in partitions {p1} and {p2} share {len(overlap)} subjects")
    
    # Check validation set overlaps
    val_sets = {}
    for p in range(num_partitions):
        val_df = result_df[(result_df['partition'] == p) & (result_df['split'] == 'val')]
        val_sets[p] = set(val_df['subject'].values)
    
    for p1 in range(num_partitions):
        for p2 in range(p1 + 1, num_partitions):
            overlap = val_sets[p1] & val_sets[p2]
            if overlap:
                warnings.append(f"Validation sets in partitions {p1} and {p2} share {len(overlap)} subjects")
    
    return {'issues': issues, 'warnings': warnings}


def print_partition_stats(result_df: pd.DataFrame, num_partitions: int, targets: Dict):
    """Print statistics about the partitions."""
    print("\n" + "="*80)
    print("PARTITION STATISTICS")
    print("="*80)
    
    for p in range(num_partitions):
        partition_df = result_df[result_df['partition'] == p]
        
        print(f"\nPartition {p}:")
        for split in ['train', 'val', 'test']:
            split_df = partition_df[partition_df['split'] == split]
            if len(split_df) == 0:
                continue
            
            clips = split_df['clip_count'].sum()
            stable = split_df['stable'].sum()
            unstable = split_df['unstable'].sum()
            total = stable + unstable
            unstable_ratio = unstable / total if total > 0 else 0
            
            target_clips = {
                'train': targets['target_train_clips'],
                'val': targets['target_val_clips'],
                'test': targets['target_test_clips']
            }[split]
            
            clip_error = abs(clips - target_clips) / target_clips * 100 if target_clips > 0 else 0
            ratio_error = abs(unstable_ratio - targets['unstable_ratio']) * 100
            
            print(f"  {split:5s}: {len(split_df):3d} subjects, {clips:5.0f} clips "
                  f"(target: {target_clips:5.0f}, error: {clip_error:5.1f}%), "
                  f"unstable ratio: {unstable_ratio:.3f} (target: {targets['unstable_ratio']:.3f}, "
                  f"error: {ratio_error:.2f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='Partition subjects into train/val/test sets across multiple partitions'
    )
    parser.add_argument('--input', type=str, default='csvs/subjects.csv',
                       help='Input CSV file with subjects')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for partition CSV files')
    parser.add_argument('--partitions', type=int, required=True,
                       help='Number of partitions (P)')
    parser.add_argument('--p-train', type=float, required=True,
                       help='Proportion of clips for training set')
    parser.add_argument('--p-val', type=float, required=True,
                       help='Proportion of clips for validation set')
    parser.add_argument('--p-test', type=float, required=True,
                       help='Proportion of clips for test set')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Validate proportions
    if abs(args.p_train + args.p_val + args.p_test - 1.0) > 1e-6:
        raise ValueError(f"Proportions must sum to 1.0, got {args.p_train + args.p_val + args.p_test}")
    
    # Read input data
    print(f"Reading data from {args.input}...")
    df = pd.read_csv(args.input, index_col=0)
    print(f"Loaded {len(df)} subjects")
    
    # Calculate targets
    targets = calculate_targets(df, args.p_train, args.p_val, args.p_test)
    print(f"\nTargets:")
    print(f"  Total clips: {targets['total_clips']}")
    print(f"  Train clips: {targets['target_train_clips']:.1f} ({args.p_train*100:.1f}%)")
    print(f"  Val clips:   {targets['target_val_clips']:.1f} ({args.p_val*100:.1f}%)")
    print(f"  Test clips:  {targets['target_test_clips']:.1f} ({args.p_test*100:.1f}%)")
    print(f"  Unstable ratio: {targets['unstable_ratio']:.3f}")
    
    # Assign subjects to partitions
    print(f"\nAssigning subjects to {args.partitions} partitions...")
    result_df = assign_subjects_to_partitions(
        df, args.partitions, args.p_train, args.p_val, args.p_test, args.seed
    )
    
    # Validate partitions
    validation = validate_partitions(result_df, args.partitions)
    
    if validation['issues']:
        print("\n❌ CRITICAL ISSUES FOUND:")
        for issue in validation['issues']:
            print(f"  - {issue}")
    else:
        print("\n✅ No test set overlaps (requirement 3 satisfied)")
    
    if validation['warnings']:
        print("\n⚠️  WARNINGS:")
        for warning in validation['warnings']:
            print(f"  - {warning}")
    
    # Print statistics
    print_partition_stats(result_df, args.partitions, targets)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Write partition CSV files
    print(f"\nWriting partition files to {args.output_dir}...")
    for p in range(args.partitions):
        partition_df = result_df[result_df['partition'] == p].copy()
        # Reorder columns: put partition and split at the end
        cols = [c for c in partition_df.columns if c not in ['partition', 'split']]
        partition_df = partition_df[cols + ['partition', 'split']]
        
        output_file = os.path.join(args.output_dir, f'partition{p}.csv')
        partition_df.to_csv(output_file, index=True)
        print(f"  Written {output_file} ({len(partition_df)} subjects)")
    
    print("\n✅ Done!")


if __name__ == '__main__':
    main()

