import pickle
import pandas as pd
import numpy as np


CLIPS       = "csvs/clips_merged.csv"
SKEL_ARRAYS_MERGED = "output_merged/skels.npz"

skels = np.load(SKEL_ARRAYS_MERGED)
clips = pd.read_csv(CLIPS, index_col=0)

def create_skel_pkl():
    test_X = []
    test_Y = []

    for _, clip in clips.iterrows():
        subj_skel = skels[clip["clip_path"]]
        # NOTE: This uses the sarcopenia-normal as Y instead of stable-unstable (beware.)
        # 0  = normal, 1 = sarcopenia
        subj_label = clip["sarcopenia-normal"] == "sarcopenia"
        test_X.append(subj_skel)
        test_Y.append(subj_label)

    return dict(
        test_X=test_X,
        test_Y=test_Y,
    )

output_path = "output_merged/merged.pkl"
out = create_skel_pkl()
for s in out:
    print(f"{s}={len(out[s])}")

print(f"Output to {output_path}")
with open(output_path, 'wb') as f:
    pickle.dump(out, f)
