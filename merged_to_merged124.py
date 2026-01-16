import pandas as pd
import numpy as np

subjects = pd.read_csv("csvs/subjects.csv", index_col=0)
IN = "/Users/aldo/Code/avlab/UPLOAD/tg_skels_166_full.npz"
OUT = "/Users/aldo/Code/avlab/UPLOAD/tg_skels_124_full.npz"
# IN = "output/merged/skels.npz"
# OUT = "output/merged_124/skels.npz"

skels = dict(np.load(IN))
skels_out = {}

not_found = []

for subject_idx, subject in subjects.iterrows():
    if subject['sarcopenia-normal'] == 'sarcopenia':
        prefix = 1
    else:
        prefix = 2

    orig_subj_idx = subject['original_subject_idx']

    mp4_key = f"{prefix}_{orig_subj_idx:03}.mp4"

    if mp4_key in skels:
        print(f"{mp4_key} found!")
        skels_out[mp4_key] = skels[mp4_key]
    else:
        print(f"WARNING: {mp4_key} not found!")


if len(not_found) == 0:
    np.savez(OUT, **skels_out)
    print(f"All {len([x for x in skels_out.keys()])} clips were found. Output skels saved to {OUT}")

