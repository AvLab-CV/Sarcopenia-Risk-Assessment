#!/usr/bin/env python3
import os
import argparse
import random
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="CSV with columns subject,stable,unstable,...")
    OUT_PATH = "fold_all.csv"

    args = parser.parse_args()
    df = pd.read_csv(args.csv)
    # build subj_stats
    subj_stats = {}
    grouped = df.groupby("subject")[["stable","unstable"]].sum().reset_index()
    for _, r in grouped.iterrows():
        subj_stats[int(r["subject"])] = {"stable": int(r["stable"]), "unstable": int(r["unstable"])}
    folds_data, desired = make_splits(subj_stats,
                                      p=(args.p_train, args.p_val, args.p_test),
                                      folds=args.folds,
                                      max_attempts=args.max_attempts,
                                      seed=args.seed)

    # write per-fold results and print summaries
    train_sets = []
    test_sets = []
    val_sets = []
    for i, fold in enumerate(folds_data):
        assign = fold["assign"]
        sums = fold["summary"]
        print(f"\nFOLD {i+1}/{len(folds_data)} summary:")
        pretty_print_summary(sums)
        
        # write subject->split CSV
        out = pd.DataFrame([
               {
                    "subject": subject,
                    "sarcopenia-normal": df[df["subject"] == subject]["sarcopenia-normal"].iloc[0],
                    "original_subject_idx": df[df["subject"] == subject]["original_subject_idx"].iloc[0],
                    "split": assign[subject],
                    "clip_count": subj_stats[subject]["stable"] + subj_stats[subject]["unstable"],
                    "stable": subj_stats[subject]["stable"],
                    "unstable": subj_stats[subject]["unstable"],
                    "path": df[df["subject"] == subject]["path"].iloc[0],
                }
                for subject in sorted(assign.keys())
            ]
        )
        out_fn = f"{OUT_DIR}/fold{i+1}_subjects.csv"
        out.to_csv(out_fn, index=False)
        print(f"  -> wrote {out_fn}")
        
        # collect sets for cross-fold comparison
        train_subs = sorted([s for s,split in assign.items() if split=="train"])
        test_subs = sorted([s for s,split in assign.items() if split=="test"])
        val_subs = sorted([s for s,split in assign.items() if split=="val"])
        train_sets.append(set(train_subs))
        test_sets.append(set(test_subs))
        val_sets.append(set(val_subs))
        print(f"  test subjects ({len(test_subs)}): {test_subs}")
        print(f"  val  subjects ({len(val_subs)}): {val_subs}")       

    sim_matrix = np.zeros((len(test_sets), len(test_sets)))
    for i in range(len(test_sets)):
        for j in range(i+1, len(test_sets)):
            similarity = len(test_sets[i] & test_sets[j]) / len(test_sets[i] | test_sets[j])
            sim_matrix[i,j] = similarity

    print()
    print("Test-set similarity matrix")
    print(sim_matrix)
    print()

    # check overlap of test and val inside each fold (should be disjoint)
    for i, (train_set, test_set, val_set) in enumerate(zip(train_sets, test_sets, val_sets)):
        inter1 = train_set & val_set
        inter2 = train_set & test_set
        inter3 = val_set   & test_set

        if inter1:
            print(f"ERROR: fold {i+1} has overlap between train and val subjects: {sorted(inter1)}")
        if inter2:
            print(f"ERROR: fold {i+1} has overlap between train and test subjects: {sorted(inter2)}")
        if inter3:
            print(f"ERROR: fold {i+1} has overlap between test and val subjects: {sorted(inter3)}")

    print("For better results, try increasing --max_attempts or changing --seed.")

if __name__ == "__main__":
    main()
