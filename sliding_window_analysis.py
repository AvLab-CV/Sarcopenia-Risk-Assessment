import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from pathlib import Path
from statsmodels.distributions.empirical_distribution import ECDF

PLOT_OUT_DIR = Path("/Users/aldo/Code/avlab/paper_typst/figures/plots/")
FOLD = 1
WINDOW_SIZE = 64
STRIDE = 8

# if a window's unstable prob > this thresh, the clip is considered unstable
UNSTABLE_WINDOW_THRESH = 0.5
# if this fraction of this subject's windows is unstable, then we predict this subject to have sarcopenia
UNSTABLE_WINDOWS_FRACTION_SARCOPENIA_THRESH = 0.5

df_original = pd.read_csv(f"./results/fold{FOLD}_sliding_window_windowsize{WINDOW_SIZE}_stride{STRIDE}.csv")
df_original['predictions'] = df_original['predictions'].map(lambda x: np.fromstring(x.strip("[ ]"), sep=' '))
largest_window = df_original['predictions'].map(lambda x: x.shape[0]).max()
# position of the beginning window `i`
window_pos = (np.arange(largest_window) * STRIDE)
df_original['window_pos'] = df_original['predictions'].map(lambda x: window_pos[:x.shape[0]])
normal_length =     df_original.loc[df_original['subject_has_sarcopenia'] == 0, 'seq_len']
sarcopenia_length = df_original.loc[df_original['subject_has_sarcopenia'] == 1, 'seq_len']

print(f"normal_length     ~ {normal_length.mean()} += {normal_length.std()}")
print(f"sarcopenia_length ~ {sarcopenia_length.mean()} += {sarcopenia_length.std()}")

def stats(df, unstable_window_thresh, unstable_windows_fraction_sarcopenia_thresh):
    """
    WARN: this modifies the passed dataframe
    """
    df['stable_count']   = df['predictions'].map(lambda x: (x <= unstable_window_thresh).sum())
    df['unstable_count'] = df['predictions'].map(lambda x: (x  > unstable_window_thresh).sum())
    df['unstable_fraction'] = df['unstable_count'] / (df['stable_count'] + df['unstable_count'])
    # we predict sarcopenia if the fraction of unstable windows is greater than this threshold
    df['sarcopenia_predicted'] = df['unstable_fraction'] > unstable_windows_fraction_sarcopenia_thresh
    correct_predictions = (df['subject_has_sarcopenia'] == df['sarcopenia_predicted']).sum()
    accuracy = correct_predictions / len(df.index)
    return accuracy
    
def plot_sliding_window(df, subject_idx, kind='bars'):
    predictions = df.loc[subject_idx, 'predictions']
    window_pos = df.loc[subject_idx, 'window_pos'].astype(np.float32)
    window_pos += WINDOW_SIZE / 2
    window_pos *= (1.0 / 30.0) # convert from frames to seconds
    seq_len = df.loc[subject_idx, 'seq_len'].astype(np.float32)
    seq_len *= (1.0 / 30.0) # convert from frames to seconds
    sarcopenia = df.loc[subject_idx, 'subject_has_sarcopenia'] == 1
    sarcopenia_predicted = df.loc[subject_idx, 'sarcopenia_predicted'] == 1
    sarcopenia_label = 'sarcopenia' if sarcopenia else 'normal'
    unstable_label = 'sarcopenia' if sarcopenia_predicted else 'normal'

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.set_title(f"Subject {subject_idx}. GT={sarcopenia_label}. Prediction={unstable_label}")
    ax.set_xlim(0.0, seq_len)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("P(unstable|window) (from model)")

    mask = predictions > UNSTABLE_WINDOW_THRESH

    ax.plot(window_pos, predictions, color='gray', linewidth=1)
    ax.axhline(UNSTABLE_WINDOW_THRESH, alpha=0.6, color=(0, 0.5, 0), linestyle='--', linewidth=1.5, label="threshold for 'unstable' label")
    if kind == 'dots':
        # stable
        ax.scatter(window_pos[~mask], predictions[~mask], color='blue', label='stable windows')
        # unstable
        ax.scatter(window_pos[mask], predictions[mask], color='red', label='unstable windows')
    elif kind == 'bars':
        # stable
        ax.bar(window_pos[~mask], predictions[~mask], width=WINDOW_SIZE / 30.0, color='blue', alpha=0.15)
        ax.scatter(window_pos[~mask], predictions[~mask], color='blue', label='stable windows')
        # unstable
        ax.bar(window_pos[mask], predictions[mask], width=WINDOW_SIZE / 30.0, color='red', alpha=0.15)
        ax.scatter(window_pos[mask], predictions[mask], color='red', label='unstable windows')
    
    # threshold line
    ax.legend()
    ax.grid()
    fig.tight_layout()
    # plt.show()
    return fig

def plot_sliding_window_all(df):
    "Plot all subjects' unstable probs (note: this reuses the modified df from the stats call)"

    os.makedirs(PLOT_OUT_DIR, exist_ok=True)
    for subject_idx in tqdm(range(len(df.index)), desc="Plotting..."):
        fig = plot_sliding_window(df, subject_idx)
        fig.savefig(PLOT_OUT_DIR / f"{subject_idx}_stability_plot.pdf")
        fig.clear()
        plt.close(fig)

def plot_cdf(series, labels, colors, variable_label):
    x = np.linspace(min([min(x) for x in series]), max([max(x) for x in series]), 400)

    fig = plt.figure(figsize=(6,4))
    ax = fig.subplots()

    for s, labels, color in zip(series, labels, colors):
        cdf   = ECDF(s)
        ax.plot(x, cdf(x), label=labels, color=color, lw=2)

    ax.set_xlabel(variable_label)
    ax.set_ylabel('Cumulative probability')
    ax.legend()
    # ax.set_title("Cumulative distribution of the instability rate for all subjects.")
    fig.tight_layout()
    return fig

# Single run with default threshold constants defined above
df = df_original.copy(deep=True)
acc = stats(df, UNSTABLE_WINDOW_THRESH, UNSTABLE_WINDOWS_FRACTION_SARCOPENIA_THRESH)
print(f"{UNSTABLE_WINDOW_THRESH=}")
print(f"{UNSTABLE_WINDOWS_FRACTION_SARCOPENIA_THRESH=}")
# TODO: plot confusion matrix with accuracy, precision, recall, f1
# We should show this. Actually, we should probably use the medical terms
# instead of the CV terms.
print(f"{acc=}")
plot_cdf(
    series=[
        df.loc[df['subject_has_sarcopenia'] == 0, 'seq_len'],
        df.loc[df['subject_has_sarcopenia'] == 1, 'seq_len'],
    ],
    labels=['Normal', 'Sarcopenia'],
    colors=['green', 'purple'],
    variable_label="Sequence length",
).savefig(PLOT_OUT_DIR / "seq_len_cdf.pdf")
plot_cdf(
    series=[
        df.loc[df['subject_has_sarcopenia'] == 0, 'unstable_fraction'],
        df.loc[df['subject_has_sarcopenia'] == 1, 'unstable_fraction'],
    ],
    labels=['Normal', 'Sarcopenia'],
    colors=['green', 'purple'],
    variable_label="Instability rate",
).savefig(PLOT_OUT_DIR / "instability_rate_cdf.pdf")
# plt.show()

# print(f"normal_unstable_fraction     ~ {normal_unstable_fraction.mean()} += {normal_unstable_fraction.std()}")
# print(f"sarcopenia_unstable_fraction ~ {sarcopenia_unstable_fraction.mean()} += {sarcopenia_unstable_fraction.std()}")

plot_sliding_window_all(df)

# Perform a grid sweep to find the best threshold (overfits to the dataset)
def sweep(SWEEP_I, SWEEP_J):
    best = None
    for i in tqdm(range(SWEEP_I)):
        for j in tqdm(range(SWEEP_J)):
            thresh1 = i / SWEEP_I
            thresh2 = j / SWEEP_J

            df = df_original.copy(True)
            acc = stats(df, thresh1, thresh2)
            if best is None or acc > best[0]:
                best = (acc, thresh1, thresh2)

    acc, thresh1, thresh2 = best
    print(f"Best threshold from {SWEEP_I}x{SWEEP_J} hyperparameter sweep")
    print(f"UNSTABLE_WINDOW_THRESH = {thresh1}")
    print(f"UNSTABLE_WINDOWS_FRACTION_SARCOPENIA_THRESH = {thresh2}")
    print(f"Accuracy = {acc}")

# sweep(100, 100)
