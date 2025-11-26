from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

SARCOPENIA_RISK_THRESHOLD = 0.2

PLOT_OUT_DIR = Path("output/dist_overlap/")
FOLD = 1
WINDOW_SIZE = 64
STRIDE = 8

df = pd.read_csv(f"./output/results/fold{FOLD}_sliding_window_windowsize{WINDOW_SIZE}_stride{STRIDE}.csv")
df['predictions'] = df['predictions'].map(lambda x: np.fromstring(x.strip("[ ]"), sep=' '))
largest_window = df['predictions'].map(lambda x: x.shape[0]).max()
# window_pos[i] = position at the beginning window i
window_pos = np.arange(largest_window) * STRIDE
df['window_pos'] = df['predictions'].map(lambda x: window_pos[:x.shape[0]])
df['stable_count']   = df['predictions'].map(lambda x: (x <= 0.5).sum())
df['unstable_count'] = df['predictions'].map(lambda x: (x  > 0.5).sum())
df['instability_rate'] = df['unstable_count'] / (df['stable_count'] + df['unstable_count'])
# # we declare sarcopenia risk if the instability than this threshold
# df['predicted_sarcopenia_risk'] = df['instability_rate'] > SARCOPENIA_RISK_THRESHOLD

fig, ax = plt.subplots()

for cls, color in zip([False, True], ['blue', 'orange']):
    data = df.loc[df['subject_has_sarcopenia'] == cls, 'instability_rate'].values
    
    # KDE with boundary correction: reflect data at 0 and 1
    reflected_data = np.concatenate([data, -data, 2 - data])
    kde = gaussian_kde(reflected_data, bw_method='scott')
    
    x = np.linspace(0, 1, 200)
    ax.plot(x, kde(x), color=color, label=cls, lw=2, alpha=0.7)

ax.set_xlabel('Instability rate')
ax.set_ylabel('Density')
ax.legend(title='Has sarcopenia')
plt.show()
