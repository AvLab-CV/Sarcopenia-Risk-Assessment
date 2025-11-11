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

def make_splits(subj_stats, p=(0.7,0.1,0.2), folds=5, max_attempts=200, seed=1234):
    all_subjs = list(subj_stats.keys())
    total_stable = sum(v["stable"] for v in subj_stats.values())
    total_unstable = sum(v["unstable"] for v in subj_stats.values())
    props = {"train": p[0], "val": p[1], "test": p[2]}
    desired = {k: {"stable": total_stable * props[k], "unstable": total_unstable * props[k]} for k in props}
    rng = random.Random(seed)

    fold_results = []

    for f in range(folds):
        best_attempt = None
        best_error = None
        # vary seed per fold to avoid identical folds
        for attempt in range(max_attempts):
            subj_order = all_subjs[:] 
            rng.shuffle(subj_order)
            assign, current = greedy_partition(subj_order, subj_stats, desired, rng)
            # compute total L1 error
            error = 0.0
            for k in desired:
                error += abs(current[k]["stable"] - desired[k]["stable"]) + abs(current[k]["unstable"] - desired[k]["unstable"])
            if best_error is None or error < best_error:
                best_error = error
                best_attempt = (assign.copy(), {k:current[k].copy() for k in current})
            # early exit if pretty good
            if error <= 0.05 * (total_stable + total_unstable):  # 5% of total clips
                break
            # shuffle seed a bit
            rng.seed(seed + f*1000 + attempt + rng.randint(0,99999))
        # finalize fold f
        assign, current = best_attempt
        # summary
        sums = summarize_assign(assign, subj_stats)
        fold_results.append({"assign": assign, "summary": sums})
    return fold_results, desired

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="CSV with columns subject,stable,unstable,...")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--p_train", type=float, default=0.7)
    parser.add_argument("--p_val", type=float, default=0.1)
    parser.add_argument("--p_test", type=float, default=0.2)
    parser.add_argument("--max_attempts", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1234)
    OUT_DIR = "folds"

    if os.path.exists(OUT_DIR):
        print(f"Output directory {OUT_DIR} already exists.")
        print("Aborting for safety.")
        exit(1)
        return

    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    # build subj_stats
    subj_stats = {}
    grouped = df.groupby("subject")[["stable","unstable"]].sum().reset_index()
    for _, r in grouped.iterrows():
        subj_stats[int(r["subject"])] = {"stable": int(r["stable"]), "unstable": int(r["unstable"])}

    # # Check if eq???
    # # print(subj_stats[])
    # for i in range(len(subj_stats)):
    #     if subj_stats[i]["stable"] != df.loc[i, "stable"]:
    #         print("NOOOOOOOOOooo")
    #         exit()

    #     if subj_stats[i]["unstable"] != df.loc[i, "unstable"]:
    #         print("NOOOOOOOOOooo 2")
    #         exit()

    folds_data, desired = make_splits(subj_stats,
                                      p=(args.p_train, args.p_val, args.p_test),
                                      folds=args.folds,
                                      max_attempts=args.max_attempts,
                                      seed=args.seed)

    # print global totals & desired
    total_stable = sum(v["stable"] for v in subj_stats.values())
    total_unstable = sum(v["unstable"] for v in subj_stats.values())
    total_clips = total_stable + total_unstable
    print("\nTOTAL clips:", total_clips, "| stable:", total_stable, "unstable:", total_unstable)
    print("Desired (approx):")
    for k in ("train","val","test"):
        print(f" {k:5} -> stable~{desired[k]['stable']:.1f} unstable~{desired[k]['unstable']:.1f}")

    os.makedirs("folds")
    print("Created directory folds")

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
                    # "path": df[df["subject"] == subject]["path"].iloc[0],
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
