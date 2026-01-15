import pandas as pd
import shutil
from pathlib import Path

df = pd.read_csv("csvs/subjects.csv", index_col=0)
print(len(df))
SRC_DIR = Path("/Users/aldo/Code/avlab/dataset/tg/")
DST_DIR = Path("/Users/aldo/Code/avlab/dataset/tg_FILTERED/")

names = []
for i, row in df.iterrows():
    sn = row["sarcopenia-normal"]
    prefix = 1 if sn == 'sarcopenia' else 2
    idx = row["original_subject_idx"]
    name = f"{prefix}_{idx:03}tg.mp4"
    names.append(name)
    # DST_DIR.mkdir(exist_ok=True)
    # shutil.copy(SRC_DIR / name, DST_DIR / name)

# df["name"] = names
# df = df[["name"]]
# df = df.sort_values("name")
# df.to_csv("csvs/subjects_malefemale_.csv")

df2 = pd.read_csv("csvs/subjects_malefemale.csv", index_col=0)
df2['prefix'] = [n[0] for n in df2['name']]
print(df2.groupby(['prefix', 'sex']).size())
