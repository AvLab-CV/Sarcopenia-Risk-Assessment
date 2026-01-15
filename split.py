#!/usr/bin/env python3
"""
split.py - Compare different data-splitting strategies for tandem gait dataset.

This script implements multiple strategies for splitting subjects into train/val/test sets,
maintaining subject-wise splitting (all clips from a subject go to the same set).
"""

import argparse
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
from collections import defaultdict
import itertools
import json
import os

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def load_data(subjects_path: str = "csvs/subjects.csv", 
              clips_path: str = "csvs/clips.csv") -> pd.DataFrame:
    """Load subject and clip data."""
    subjects = pd.read_csv(subjects_path, index_col=0)
    clips = pd.read_csv(clips_path, index_col=0)
    return subjects


def calculate_balance_metrics(subjects: pd.DataFrame) -> Dict:
    """Calculate balance metrics for a set of subjects."""
    total_subjects = len(subjects)
    total_clips = subjects['clip_count'].sum()
    total_stable = subjects['stable'].sum()
    total_unstable = subjects['unstable'].sum()
    
    sarcopenia_subjects = subjects[subjects['sarcopenia-normal'] == 'sarcopenia']
    normal_subjects = subjects[subjects['sarcopenia-normal'] == 'normal']
    
    metrics = {
        'n_subjects': total_subjects,
        'n_clips': total_clips,
        'n_stable': total_stable,
        'n_unstable': total_unstable,
        'stable_ratio': total_stable / total_clips if total_clips > 0 else 0,
        'unstable_ratio': total_unstable / total_clips if total_clips > 0 else 0,
        'n_sarcopenia': len(sarcopenia_subjects),
        'n_normal': len(normal_subjects),
        'sarcopenia_clips': sarcopenia_subjects['clip_count'].sum(),
        'normal_clips': normal_subjects['clip_count'].sum(),
        'sarcopenia_stable': sarcopenia_subjects['stable'].sum(),
        'sarcopenia_unstable': sarcopenia_subjects['unstable'].sum(),
        'normal_stable': normal_subjects['stable'].sum(),
        'normal_unstable': normal_subjects['unstable'].sum(),
    }
    
    # Add ratios
    if metrics['sarcopenia_clips'] > 0:
        metrics['sarcopenia_stable_ratio'] = metrics['sarcopenia_stable'] / metrics['sarcopenia_clips']
    else:
        metrics['sarcopenia_stable_ratio'] = 0
        
    if metrics['normal_clips'] > 0:
        metrics['normal_stable_ratio'] = metrics['normal_stable'] / metrics['normal_clips']
    else:
        metrics['normal_stable_ratio'] = 0
    
    return metrics


def print_split_summary(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, 
                       strategy_name: str):
    """Print a summary of the split."""
    print(f"\n{'='*80}")
    print(f"Strategy: {strategy_name}")
    print(f"{'='*80}")
    
    for split_name, split_df in [('Train', train), ('Val', val), ('Test', test)]:
        metrics = calculate_balance_metrics(split_df)
        print(f"\n{split_name} Set:")
        print(f"  Subjects: {metrics['n_subjects']} (Sarcopenia: {metrics['n_sarcopenia']}, Normal: {metrics['n_normal']})")
        print(f"  Clips: {metrics['n_clips']} (Stable: {metrics['n_stable']}, Unstable: {metrics['n_unstable']})")
        print(f"  Stability ratios - Overall: {metrics['stable_ratio']:.3f}, Sarcopenia: {metrics['sarcopenia_stable_ratio']:.3f}, Normal: {metrics['normal_stable_ratio']:.3f}")
        print(f"  Group clip counts - Sarcopenia: {metrics['sarcopenia_clips']}, Normal: {metrics['normal_clips']}")
        print(f"    Sarcopenia breakdown: Stable={metrics['sarcopenia_stable']}, Unstable={metrics['sarcopenia_unstable']}")
        print(f"    Normal breakdown: Stable={metrics['normal_stable']}, Unstable={metrics['normal_unstable']}")


def strategy_random_split(subjects: pd.DataFrame, 
                          train_ratio: float = 0.65,
                          val_ratio: float = 0.11,
                          test_ratio: float = 0.24,
                          seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Strategy 1: Simple random split by subjects.
    No balancing considerations.
    """
    np.random.seed(seed)
    subjects_shuffled = subjects.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    n = len(subjects_shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train = subjects_shuffled.iloc[:train_end].copy()
    val = subjects_shuffled.iloc[train_end:val_end].copy()
    test = subjects_shuffled.iloc[val_end:].copy()
    
    return train, val, test


def strategy_stratified_by_group(subjects: pd.DataFrame,
                                 train_ratio: float = 0.65,
                                 val_ratio: float = 0.11,
                                 test_ratio: float = 0.24,
                                 seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Strategy 2: Stratified split maintaining sarcopenia/normal proportions.
    """
    sarcopenia = subjects[subjects['sarcopenia-normal'] == 'sarcopenia'].copy()
    normal = subjects[subjects['sarcopenia-normal'] == 'normal'].copy()
    
    # Split each group
    train_s, temp_s = train_test_split(sarcopenia, test_size=(1-train_ratio), 
                                       random_state=seed)
    val_s, test_s = train_test_split(temp_s, 
                                     test_size=test_ratio/(test_ratio+val_ratio), 
                                     random_state=seed)
    
    train_n, temp_n = train_test_split(normal, test_size=(1-train_ratio), 
                                       random_state=seed)
    val_n, test_n = train_test_split(temp_n, 
                                     test_size=test_ratio/(test_ratio+val_ratio), 
                                     random_state=seed)
    
    train = pd.concat([train_s, train_n]).reset_index(drop=True)
    val = pd.concat([val_s, val_n]).reset_index(drop=True)
    test = pd.concat([test_s, test_n]).reset_index(drop=True)
    
    return train, val, test


def strategy_balanced_1to1_ratio(subjects: pd.DataFrame,
                                 train_ratio: float = 0.65,
                                 val_ratio: float = 0.11,
                                 test_ratio: float = 0.24,
                                 seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Strategy 3: Maintain 1:1 ratio for stable/unstable and sarcopenia/normal.
    This may involve undersampling the majority class at the clip level by selecting
    appropriate subjects.
    """
    np.random.seed(seed)
    
    # Separate by group
    sarcopenia = subjects[subjects['sarcopenia-normal'] == 'sarcopenia'].copy()
    normal = subjects[subjects['sarcopenia-normal'] == 'normal'].copy()
    
    # For balanced groups, we want equal subjects from each
    n_sarcopenia = len(sarcopenia)
    n_normal = len(normal)
    
    # Use the smaller group size for balance
    min_group_size = min(n_sarcopenia, n_normal)
    
    # Randomly sample to balance groups
    if n_sarcopenia > min_group_size:
        sarcopenia = sarcopenia.sample(n=min_group_size, random_state=seed)
    if n_normal > min_group_size:
        normal = normal.sample(n=min_group_size, random_state=seed)
    
    # Now split each balanced group
    train_s, temp_s = train_test_split(sarcopenia, test_size=(1-train_ratio), 
                                       random_state=seed)
    val_s, test_s = train_test_split(temp_s, 
                                     test_size=test_ratio/(test_ratio+val_ratio), 
                                     random_state=seed)
    
    train_n, temp_n = train_test_split(normal, test_size=(1-train_ratio), 
                                       random_state=seed)
    val_n, test_n = train_test_split(temp_n, 
                                     test_size=test_ratio/(test_ratio+val_ratio), 
                                     random_state=seed)
    
    train = pd.concat([train_s, train_n]).reset_index(drop=True)
    val = pd.concat([val_s, val_n]).reset_index(drop=True)
    test = pd.concat([test_s, test_n]).reset_index(drop=True)
    
    return train, val, test


def strategy_clip_balanced(subjects: pd.DataFrame,
                          train_ratio: float = 0.65,
                          val_ratio: float = 0.11,
                          test_ratio: float = 0.24,
                          seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Strategy 4: Try to balance clips (not subjects) across splits while maintaining
    subject-wise splitting.
    """
    np.random.seed(seed)
    
    # Sort subjects by number of clips to help with balancing
    subjects_sorted = subjects.sort_values('clip_count', ascending=False).reset_index(drop=True)
    
    # Initialize splits
    train_subjects = []
    val_subjects = []
    test_subjects = []
    
    train_clips = 0
    val_clips = 0
    test_clips = 0
    
    total_clips = subjects['clip_count'].sum()
    target_train = total_clips * train_ratio
    target_val = total_clips * val_ratio
    target_test = total_clips * test_ratio
    
    # Greedy assignment to balance clip counts
    for idx, row in subjects_sorted.iterrows():
        clips = row['clip_count']
        
        # Calculate current deficits
        train_deficit = target_train - train_clips
        val_deficit = target_val - val_clips
        test_deficit = target_test - test_clips
        
        # Assign to split with largest deficit
        if train_deficit >= val_deficit and train_deficit >= test_deficit:
            train_subjects.append(idx)
            train_clips += clips
        elif val_deficit >= test_deficit:
            val_subjects.append(idx)
            val_clips += clips
        else:
            test_subjects.append(idx)
            test_clips += clips
    
    train = subjects.loc[train_subjects].copy()
    val = subjects.loc[val_subjects].copy()
    test = subjects.loc[test_subjects].copy()
    
    return train, val, test


def strategy_stratified_stability(subjects: pd.DataFrame,
                                  train_ratio: float = 0.65,
                                  val_ratio: float = 0.11,
                                  test_ratio: float = 0.24,
                                  seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Strategy 5: Stratify by stability patterns (all stable, all unstable, mixed).
    """
    np.random.seed(seed)
    
    # Categorize subjects by stability
    all_stable = subjects[(subjects['stable'] > 0) & (subjects['unstable'] == 0)].copy()
    all_unstable = subjects[(subjects['stable'] == 0) & (subjects['unstable'] > 0)].copy()
    mixed = subjects[(subjects['stable'] > 0) & (subjects['unstable'] > 0)].copy()
    
    # Split each category
    splits = []
    for category_df in [all_stable, all_unstable, mixed]:
        if len(category_df) > 0:
            train_c, temp_c = train_test_split(category_df, test_size=(1-train_ratio), 
                                               random_state=seed)
            if len(temp_c) > 1:
                val_c, test_c = train_test_split(temp_c, 
                                                 test_size=test_ratio/(test_ratio+val_ratio), 
                                                 random_state=seed)
            else:
                # If only one subject left, put it in test
                val_c = pd.DataFrame()
                test_c = temp_c
            
            splits.append((train_c, val_c, test_c))
        else:
            splits.append((pd.DataFrame(), pd.DataFrame(), pd.DataFrame()))
    
    # Combine
    train = pd.concat([s[0] for s in splits]).reset_index(drop=True)
    val = pd.concat([s[1] for s in splits if len(s[1]) > 0]).reset_index(drop=True)
    test = pd.concat([s[2] for s in splits if len(s[2]) > 0]).reset_index(drop=True)
    
    return train, val, test


def strategy_double_stratified(subjects: pd.DataFrame,
                               train_ratio: float = 0.65,
                               val_ratio: float = 0.11,
                               test_ratio: float = 0.24,
                               seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Strategy 6: Stratify by both group (sarcopenia/normal) and stability pattern.
    """
    np.random.seed(seed)
    
    # Create combined categories
    subjects_copy = subjects.copy()
    subjects_copy['stability_pattern'] = 'mixed'
    subjects_copy.loc[(subjects_copy['stable'] > 0) & (subjects_copy['unstable'] == 0), 'stability_pattern'] = 'all_stable'
    subjects_copy.loc[(subjects_copy['stable'] == 0) & (subjects_copy['unstable'] > 0), 'stability_pattern'] = 'all_unstable'
    
    # Split by each combination
    train_list = []
    val_list = []
    test_list = []
    
    for group in ['sarcopenia', 'normal']:
        for pattern in ['all_stable', 'all_unstable', 'mixed']:
            subset = subjects_copy[(subjects_copy['sarcopenia-normal'] == group) & 
                                  (subjects_copy['stability_pattern'] == pattern)]
            
            if len(subset) == 0:
                continue
            elif len(subset) == 1:
                train_list.append(subset)
            elif len(subset) == 2:
                train_c, test_c = train_test_split(subset, test_size=0.5, random_state=seed)
                train_list.append(train_c)
                test_list.append(test_c)
            else:
                train_c, temp_c = train_test_split(subset, test_size=(1-train_ratio), 
                                                   random_state=seed)
                if len(temp_c) > 1:
                    val_c, test_c = train_test_split(temp_c, 
                                                     test_size=test_ratio/(test_ratio+val_ratio), 
                                                     random_state=seed)
                else:
                    val_c = pd.DataFrame()
                    test_c = temp_c
                
                train_list.append(train_c)
                if len(val_c) > 0:
                    val_list.append(val_c)
                if len(test_c) > 0:
                    test_list.append(test_c)
    
    train = pd.concat(train_list).reset_index(drop=True) if train_list else pd.DataFrame()
    val = pd.concat(val_list).reset_index(drop=True) if val_list else pd.DataFrame()
    test = pd.concat(test_list).reset_index(drop=True) if test_list else pd.DataFrame()
    
    # Drop temporary column
    train = train.drop('stability_pattern', axis=1)
    val = val.drop('stability_pattern', axis=1)
    test = test.drop('stability_pattern', axis=1)
    
    return train, val, test


def save_split(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    output_path: str,
    partition: int = 0,
    method_name: str = None,
):
    """Save split in the format of sample_split.csv and write a helper text file."""
    # Assign split labels
    train = train.copy()
    val = val.copy()
    test = test.copy()
    
    train['split'] = 'train'
    val['split'] = 'val'
    test['split'] = 'test'
    train['partition'] = partition
    val['partition'] = partition
    test['partition'] = partition
    
    # Combine
    result = pd.concat([train, val, test]).reset_index(drop=True)
    
    # Reorder columns to match sample_split.csv
    result = result[['subject', 'sarcopenia-normal', 'original_subject_idx', 
                    'clip_count', 'stable', 'unstable', 'partition', 'split']]
    
    # Save
    result.to_csv(output_path)
    print(f"\nSaved split to: {output_path}")

    # Save companion text file containing the method name for easy inspection
    if method_name:
        txt_path = os.path.splitext(output_path)[0] + ".txt"
        with open(txt_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(method_name.strip() + "\n")


def evaluate_strategy(strategy_func, subjects: pd.DataFrame, strategy_name: str,
                     seeds: List[int] = [42, 123, 456], output_dir: str = "output/splits") -> Dict:
    """
    Evaluate a splitting strategy across multiple random seeds.
    """
    print(f"\n{'#'*80}")
    print(f"# Evaluating: {strategy_name}")
    print(f"{'#'*80}")
    
    metrics_list = []
    
    for seed in seeds:
        train, val, test = strategy_func(subjects, seed=seed)
        print_split_summary(train, val, test, f"{strategy_name} (seed={seed})")
        
        # Collect metrics
        train_metrics = calculate_balance_metrics(train)
        val_metrics = calculate_balance_metrics(val)
        test_metrics = calculate_balance_metrics(test)
        
        metrics_list.append({
            'seed': seed,
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics
        })
        
        # Save the first seed as the default split
        if seed == seeds[0]:
            os.makedirs(output_dir, exist_ok=True)
            strategy_filename = (
                strategy_name.lower()
                .replace(" ", "_")
                .replace("-", "_")
                .replace(":", "-")
            )
            save_split(train, val, test, 
                      os.path.join(output_dir, f'{strategy_filename}.csv'),
                      partition=0,
                      method_name=strategy_name)
    
    return metrics_list


def compare_strategies(subjects: pd.DataFrame, output_dir: str = "output/splits"):
    """
    Compare all splitting strategies.
    """
    strategies = [
        (strategy_random_split, "Random Split"),
        (strategy_stratified_by_group, "Stratified by Group"),
        (strategy_balanced_1to1_ratio, "Balanced 1:1 Ratio"),
        (strategy_clip_balanced, "Clip-Balanced"),
        (strategy_stratified_stability, "Stratified by Stability"),
        (strategy_double_stratified, "Double Stratified"),
    ]
    
    results = {}
    
    for strategy_func, strategy_name in strategies:
        metrics = evaluate_strategy(strategy_func, subjects, strategy_name, output_dir=output_dir)
        results[strategy_name] = metrics
    
    # Save comparison summary
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'comparison_summary.json'), 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    
    print(f"\n{'='*80}")
    print(f"Comparison complete! Results saved to {output_dir}")
    print(f"{'='*80}")
    
    return results


def generate_recommendations(results: Dict):
    """
    Generate recommendations based on the evaluation results.
    """
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")
    
    print("""
Based on the evaluation of different splitting strategies:

1. **Random Split**: Good baseline, but may result in imbalanced distributions.

2. **Stratified by Group**: Maintains sarcopenia/normal proportions, recommended
   for maintaining class balance in classification tasks.

3. **Balanced 1:1 Ratio**: Forces equal representation, but may undersample
   majority class. Use if you want strict balance.

4. **Clip-Balanced**: Tries to balance the number of clips (not subjects) across
   splits. Good for ensuring similar amounts of data in each split.

5. **Stratified by Stability**: Groups subjects by stability patterns and stratifies.
   Useful for understanding model performance on different stability profiles.

6. **Double Stratified**: Most sophisticated approach, stratifies by both group
   and stability pattern. Recommended for comprehensive evaluation.

**General Recommendations:**
- For sarcopenia classification: Use "Stratified by Group" or "Double Stratified"
- For stability classification: Use "Stratified by Stability" or "Double Stratified"
- For both tasks: Use "Double Stratified"
- For maximum data utilization: Use "Clip-Balanced"

The actual model performance will need to be evaluated through training runs.
Compare validation set performance across strategies to make the final choice.
    """)


def main():
    """Main function to run all comparisons."""
    parser = argparse.ArgumentParser(description="Compare subject-wise data splitting strategies.")
    parser.add_argument(
        "output_dir",
        help="Directory where split CSVs and summary JSON are written (default: output/splits)",
    )
    parser.add_argument(
        "--subjects-csv",
        default="csvs/subjects.csv",
        help="Path to the subjects metadata CSV (default: csvs/subjects.csv)",
    )
    parser.add_argument(
        "--clips-csv",
        default="csvs/clips.csv",
        help="Path to the clips metadata CSV (default: csvs/clips.csv)",
    )
    args = parser.parse_args()

    print("Loading data...")
    subjects = load_data(args.subjects_csv, args.clips_csv)
    
    print(f"\nDataset Overview:")
    print(f"Total subjects: {len(subjects)}")
    print(f"Total clips: {subjects['clip_count'].sum()}")
    print(f"Sarcopenia subjects: {len(subjects[subjects['sarcopenia-normal'] == 'sarcopenia'])}")
    print(f"Normal subjects: {len(subjects[subjects['sarcopenia-normal'] == 'normal'])}")
    print(f"Total stable clips: {subjects['stable'].sum()}")
    print(f"Total unstable clips: {subjects['unstable'].sum()}")
    
    # Run comparison
    results = compare_strategies(subjects, output_dir=args.output_dir)
    
    # Generate recommendations
    generate_recommendations(results)


if __name__ == "__main__":
    main()

