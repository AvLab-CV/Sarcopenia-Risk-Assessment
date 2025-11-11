import pandas as pd
from pathlib import Path

CLS_SARCOPENIA = 1
CLS_NORMAL = 2

VIDEO_PATH = Path("/Users/aldo/Code/avlab/dataset/merged/")
samples_paths = list(VIDEO_PATH.iterdir())

data = []
for path in samples_paths:
    parts = path.stem.split("_")
    class_sarcopenia_normal = int(parts[0])
    subject = int(parts[1])

    if class_sarcopenia_normal == CLS_SARCOPENIA:
        label_sarcopenia_normal = "sarcopenia"
    if class_sarcopenia_normal == CLS_NORMAL:
        label_sarcopenia_normal = "normal"

    full_path = path.name
    data.append((subject, label_sarcopenia_normal, full_path))

df = pd.DataFrame(data, columns=["subject", "sarcopenia-normal", "clip_path"])

df["original_subject_idx"] = df["subject"]
df.loc[df["sarcopenia-normal"] == "sarcopenia", "subject"] += 1000
# print(df.sort_values(by="subject"))
df.to_csv("csvs/clips_merged.csv")
    
# # Collapse clips into subjects
# df = df.groupby(
#     ['subject', 'sarcopenia-normal', 'original_subject_idx']
# ).size().unstack(fill_value=0).reset_index()
# # df['subject'] = df.index
# # Reorder
# df = df[[
#     'subject',
#     'sarcopenia-normal',
#     'original_subject_idx',
#     'clip_path'
# ]]
# df.to_csv("csvs/subjects_merged.csv")
