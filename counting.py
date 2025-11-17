import pandas as pd
from pathlib import Path

CLS_STABLE = 0
CLS_UNSTABLE = 1
CLS_SARCOPENIA = 1
CLS_NORMAL = 2

VIDEO_PATH = Path("/Users/aldo/Code/avlab/dataset/all_124_nosub1")
samples_paths = [path for path in VIDEO_PATH.iterdir() if path.suffix == ".mp4"]
samples_paths.sort()

data = []
for path in samples_paths:
    parts = path.stem.split("_")
    # print(parts)
    class_sarcopenia_normal = int(parts[0])
    subject = int(parts[1])
    # _length = parts[2]
    class_stable_unstable = int(parts[3])

    if class_sarcopenia_normal == CLS_SARCOPENIA:
        label_sarcopenia_normal = "sarcopenia"
    if class_sarcopenia_normal == CLS_NORMAL:
        label_sarcopenia_normal = "normal"

    if class_stable_unstable == CLS_STABLE:
        label_stable_unstable = "stable"
    if class_stable_unstable == CLS_UNSTABLE:
        label_stable_unstable = "unstable"

    full_path = path.name
    data.append((subject, label_stable_unstable, label_sarcopenia_normal, full_path))

df = pd.DataFrame(data, columns=["subject", "stable-unstable", "sarcopenia-normal", "clip_path"])

df["original_subject_idx"] = df["subject"]
df.loc[df["sarcopenia-normal"] == "sarcopenia", "subject"] += 1000
# df['subject'] = df.index
df.to_csv("csvs/clips.csv")
    
# Collapse clips into subjects
df = df.groupby(
    ['subject', 'sarcopenia-normal', 'original_subject_idx', 'stable-unstable']
).size().unstack(fill_value=0).reset_index()
df['clip_count'] = df['stable'] + df['unstable']
# df['subject'] = df.index
# Reorder
df = df[[
    'subject',
    'sarcopenia-normal',
    'original_subject_idx',
    'clip_count',
    'stable',
    'unstable',
]]
df.to_csv("csvs/subjects.csv")

# print(f"{df.loc[df['sarcopenia-normal'] == 'normal', 'unstable'].mean()}")
# print(f"{df.loc[df['sarcopenia-normal'] == 'sarcopenia', 'unstable'].mean()}")
# P_unstable = (df["stable-unstable"] == "unstable").mean()
# P_sarcopenia = (df["sarcopenia-normal"] == "sarcopenia").mean()
# ct = pd.crosstab(df["stable-unstable"], df["sarcopenia-normal"], normalize="index")
# P_sarcopenia_given_unstable = ct.loc["unstable", "sarcopenia"]
# print((df["stable-unstable"] == "stable").sum())
# print((df["stable-unstable"] == "unstable").sum())
# print((df["sarcopenia-normal"] == "sarcopenia").sum())
# print((df["sarcopenia-normal"] == "normal").sum())
# print(f"{P_unstable=:.3f}")
# print(f"{P_sarcopenia=:.3f}")
# print(f"{P_sarcopenia_given_unstable=:.3f}")
