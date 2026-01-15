import pandas as pd

# Load subjects and clips metadata
subjects_df = pd.read_csv("csvs/subjects.csv", index_col=0)
clips_df = pd.read_csv("csvs/clips.csv", index_col=0)

# Total number of clips (each row in clips.csv is a clip)
clip_count = len(clips_df)

# Stable / unstable clip counts
stable_mask = clips_df["stable-unstable"] == "stable"
unstable_mask = clips_df["stable-unstable"] == "unstable"

stable_clip_count = stable_mask.sum()
unstable_clip_count = unstable_mask.sum()

stable_clip_ratio = stable_clip_count / clip_count if clip_count else 0
unstable_clip_ratio = unstable_clip_count / clip_count if clip_count else 0

# Total number of subjects
subject_count = len(subjects_df)
mean_clips_per_subject = clip_count / subject_count

# Range of clips per subject (min / max across subjects)
min_clips_per_subject = subjects_df["clip_count"].min()
max_clips_per_subject = subjects_df["clip_count"].max()

# Subjects with and without sarcopenia
sarcopenia_mask = subjects_df["sarcopenia-normal"] == "sarcopenia"
normal_mask = subjects_df["sarcopenia-normal"] == "normal"

subject_sarcopenia_count = sarcopenia_mask.sum()
subject_normal_count = normal_mask.sum()

subject_sarcopenia_ratio = subject_sarcopenia_count / subject_count if subject_count else 0
subject_normal_ratio = subject_normal_count / subject_count if subject_count else 0

print(f"Total clips: {clip_count}")
print(f"Total subjects: {subject_count}")
print(f"Mean clips per subject: {mean_clips_per_subject:.3f}")
print(
    f"Range of clips per subject: min={min_clips_per_subject}, "
    f"max={max_clips_per_subject}"
)
print(
    f"Subjects with sarcopenia: {subject_sarcopenia_count} "
    f"({subject_sarcopenia_ratio:.3f} of total)"
)
print(
    f"Subjects without sarcopenia (normal): {subject_normal_count} "
    f"({subject_normal_ratio:.3f} of total)"
)
print(f"Stable clips: {stable_clip_count} ({stable_clip_ratio:.3f} of total)")
print(f"Unstable clips: {unstable_clip_count} ({unstable_clip_ratio:.3f} of total)")
