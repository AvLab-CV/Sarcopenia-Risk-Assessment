#!/usr/bin/env python3
import os
import argparse
import random
import pandas as pd
import numpy as np

def summarize_assign(assign_map, subj_stats):
    # assign_map: {subject: split_name}
    sums = {k: {"stable":0, "unstable":0, "clips":0, "subjects":0} for k in ("train","val","test")}
    for s, row in subj_stats.items():
        split = assign_map[s]
        sums[split]["stable"] += row["stable"]
        sums[split]["unstable"] += row["unstable"]
        sums[split]["clips"] += row["stable"] + row["unstable"]
        sums[split]["subjects"] += 1
    return sums

def pretty_print_summary(sums):
    for k in ("train","val","test"):
        st = sums[k]["stable"]
        un = sums[k]["unstable"]
        clips = sums[k]["clips"]
        subs = sums[k]["subjects"]
        ratio = f"{st}:{un}" if (st+un)>0 else "0:0"
        print(f"  {k:5} | subjects={subs:2d} | clips={clips:3d} | stable:unstable={ratio} | stable%={st/(clips+1e-9):.2f}")

def greedy_partition(subjects, subj_stats, desired, rng):
    # subjects: list of subject ids in some random order
    # subj_stats: dict subj -> {"stable": int, "unstable": int}
    # desired: dict split -> {"stable": float, "unstable": float}
    # returns assign_map
    current = {k: {"stable":0.0, "unstable":0.0} for k in desired}
    assign = {}
    for s in subjects:
        s_st = subj_stats[s]["stable"]
        s_un = subj_stats[s]["unstable"]
        best_split = None
        best_cost = None
        # evaluate adding this subject to each split; pick one that minimizes L1 error to desired
        for k in desired:
            # compute total L1 distance across splits after hypothetical assignment
            cost = 0.0
            for kk in desired:
                st = current[kk]["stable"] + (s_st if kk==k else 0.0)
                un = current[kk]["unstable"] + (s_un if kk==k else 0.0)
                cost += abs(st - desired[kk]["stable"]) + abs(un - desired[kk]["unstable"])
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_split = k
        assign[s] = best_split
        current[best_split]["stable"] += s_st
        current[best_split]["unstable"] += s_un
    return assign, current

def compute_overlap(assign1, assign2, split1='test', split2='test'):
    """Compute overlap between two assignments for given splits"""
    set1 = {s for s, split in assign1.items() if split == split1}
    set2 = {s for s, split in assign2.items() if split == split2}
    if len(set1) == 0 or len(set2) == 0:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)

def make_diverse_splits(subj_stats, p=(0.7,0.1,0.2), partitions=5, max_attempts=500, 
                        max_overlap=0.15, seed=1234):
    """
    Create partitions with explicitly diverse test/val sets.
    max_overlap: maximum Jaccard similarity allowed between test sets of different partitions
    """
    all_subjs = list(subj_stats.keys())
    total_stable = sum(v["stable"] for v in subj_stats.values())
    total_unstable = sum(v["unstable"] for v in subj_stats.values())
    props = {"train": p[0], "val": p[1], "test": p[2]}
    desired = {k: {"stable": total_stable * props[k], "unstable": total_unstable * props[k]} for k in props}
    rng = random.Random(seed)

    partition_results = []
    previous_partitions = []  # Store previous partition assignments

    for f in range(partitions):
        print(f"\nSearching for partition {f+1}...")
        best_attempt = None
        best_score = None
        attempts_made = 0
        
        for attempt in range(max_attempts):
            attempts_made += 1
            subj_order = all_subjs[:] 
            rng.shuffle(subj_order)
            assign, current = greedy_partition(subj_order, subj_stats, desired, rng)
            
            # Compute balance error (L1)
            balance_error = 0.0
            for k in desired:
                balance_error += abs(current[k]["stable"] - desired[k]["stable"]) + abs(current[k]["unstable"] - desired[k]["unstable"])
            
            # Compute diversity penalty: overlap with previous partitions
            diversity_penalty = 0.0
            max_test_overlap = 0.0
            max_val_overlap = 0.0
            
            if previous_partitions:
                for prev_assign in previous_partitions:
                    test_overlap = compute_overlap(assign, prev_assign, 'test', 'test')
                    val_overlap = compute_overlap(assign, prev_assign, 'val', 'val')
                    max_test_overlap = max(max_test_overlap, test_overlap)
                    max_val_overlap = max(max_val_overlap, val_overlap)
                    # Heavy penalty for overlapping test/val sets
                    diversity_penalty += test_overlap * 1000.0  # Very high weight
                    diversity_penalty += val_overlap * 500.0    # High weight
            
            # Combined score: balance error + diversity penalty
            total_clips = total_stable + total_unstable
            normalized_balance = balance_error / total_clips
            score = normalized_balance + diversity_penalty
            
            if best_score is None or score < best_score:
                best_score = score
                best_attempt = (assign.copy(), {k:current[k].copy() for k in current}, 
                               max_test_overlap, max_val_overlap, normalized_balance)
            
            # Early exit if we found a good diverse solution
            if max_test_overlap <= max_overlap and max_val_overlap <= max_overlap and normalized_balance <= 0.05:
                print(f"  Found good solution at attempt {attempts_made}")
                break
            
            # Vary seed
            if attempt % 50 == 0 and attempt > 0:
                print(f"  Attempt {attempt}: best test_overlap={max_test_overlap:.3f}, val_overlap={max_val_overlap:.3f}")
            rng.seed(seed + f*10000 + attempt + rng.randint(0,99999))
        
        # Finalize partition f
        assign, current, test_overlap, val_overlap, balance = best_attempt
        print(f"  Final: test_overlap={test_overlap:.3f}, val_overlap={val_overlap:.3f}, balance={balance:.3f}")
        
        previous_partitions.append(assign)
        
        # Summary
        sums = summarize_assign(assign, subj_stats)
        partition_results.append({"assign": assign, "summary": sums})
    
    return partition_results, desired

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="CSV with columns subject,stable,unstable,...")
    parser.add_argument("out_dir", type=str)
    parser.add_argument("--partitions", type=int, default=5)
    parser.add_argument("--p_train", type=float, default=0.7)
    parser.add_argument("--p_val", type=float, default=0.1)
    parser.add_argument("--p_test", type=float, default=0.2)
    parser.add_argument("--max_attempts", type=int, default=500)
    parser.add_argument("--max_overlap", type=float, default=0.15, 
                       help="Maximum allowed Jaccard overlap between test sets (0-1)")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    OUT_DIR = args.out_dir

    if os.path.exists(OUT_DIR):
        print(f"Output directory {OUT_DIR} already exists.")
        print("Aborting for safety.")
        exit(1)
        return

    df = pd.read_csv(args.csv)
    # build subj_stats
    subj_stats = {}
    grouped = df.groupby("subject")[["stable","unstable"]].sum().reset_index()
    for _, r in grouped.iterrows():
        subj_stats[int(r["subject"])] = {"stable": int(r["stable"]), "unstable": int(r["unstable"])}

    partitions_data, desired = make_diverse_splits(subj_stats,
                                              p=(args.p_train, args.p_val, args.p_test),
                                              partitions=args.partitions,
                                              max_attempts=args.max_attempts,
                                              max_overlap=args.max_overlap,
                                              seed=args.seed)

    # print global totals & desired
    total_stable = sum(v["stable"] for v in subj_stats.values())
    total_unstable = sum(v["unstable"] for v in subj_stats.values())
    total_clips = total_stable + total_unstable
    print("\nTOTAL clips:", total_clips, "| stable:", total_stable, "unstable:", total_unstable)
    print("Desired (approx):")
    for k in ("train","val","test"):
        print(f" {k:5} -> stable~{desired[k]['stable']:.1f} unstable~{desired[k]['unstable']:.1f}")

    os.makedirs(OUT_DIR)
    print(f"\nCreated directory {OUT_DIR}")

    # write per-partition results and print summaries
    train_sets = []
    test_sets = []
    val_sets = []
    for i, partition in enumerate(partitions_data):
        assign = partition["assign"]
        sums = partition["summary"]
        print(f"\npartition {i+1}/{len(partitions_data)} summary:")
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
                }
                for subject in sorted(assign.keys())
            ]
        )
        out_fn = f"{OUT_DIR}/partition{i+1}.csv"
        out.to_csv(out_fn)
        print(f"  -> wrote {out_fn}")
        
        # collect sets for cross-partition comparison
        train_subs = sorted([s for s,split in assign.items() if split=="train"])
        test_subs = sorted([s for s,split in assign.items() if split=="test"])
        val_subs = sorted([s for s,split in assign.items() if split=="val"])
        train_sets.append(set(train_subs))
        test_sets.append(set(test_subs))
        val_sets.append(set(val_subs))
        print(f"  test subjects ({len(test_subs)}): {test_subs}")
        print(f"  val  subjects ({len(val_subs)}): {val_subs}")       

    # Compute overlap matrices
    print("\n" + "="*60)
    print("TEST SET OVERLAP MATRIX (IoU/Jaccard similarity)")
    print("="*60)
    test_matrix = np.zeros((len(test_sets), len(test_sets)))
    for i in range(len(test_sets)):
        for j in range(len(test_sets)):
            if i != j:
                similarity = len(test_sets[i] & test_sets[j]) / len(test_sets[i] | test_sets[j])
                test_matrix[i,j] = similarity
    print(test_matrix)
    print(f"\nAverage test set overlap: {np.mean(test_matrix[test_matrix > 0]):.3f}")
    print(f"Max test set overlap: {np.max(test_matrix):.3f}")

    print("\n" + "="*60)
    print("VAL SET OVERLAP MATRIX (Jaccard similarity)")
    print("="*60)
    val_matrix = np.zeros((len(val_sets), len(val_sets)))
    for i in range(len(val_sets)):
        for j in range(len(val_sets)):
            if i != j:
                similarity = len(val_sets[i] & val_sets[j]) / len(val_sets[i] | val_sets[j])
                val_matrix[i,j] = similarity
    print(val_matrix)
    print(f"\nAverage val set overlap: {np.mean(val_matrix[val_matrix > 0]):.3f}")
    print(f"Max val set overlap: {np.max(val_matrix):.3f}")

    # check overlap within each partition (should be disjoint)
    print("\n" + "="*60)
    print("CHECKING WITHIN-partition OVERLAPS (should all be empty)")
    print("="*60)
    for i, (train_set, test_set, val_set) in enumerate(zip(train_sets, test_sets, val_sets)):
        inter1 = train_set & val_set
        inter2 = train_set & test_set
        inter3 = val_set   & test_set

        if inter1:
            print(f"ERROR: partition {i+1} has overlap between train and val subjects: {sorted(inter1)}")
        if inter2:
            print(f"ERROR: partition {i+1} has overlap between train and test subjects: {sorted(inter2)}")
        if inter3:
            print(f"ERROR: partition {i+1} has overlap between test and val subjects: {sorted(inter3)}")
    
    print("\n" + "="*60)
    print(f"If overlap is too high, try:")
    print(f"  - Increasing --max_attempts (current: {args.max_attempts})")
    print(f"  - Decreasing --max_overlap (current: {args.max_overlap})")
    print(f"  - Changing --seed (current: {args.seed})")

if __name__ == "__main__":
    main()
