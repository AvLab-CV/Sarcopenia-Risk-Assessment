import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import roc_curve, auc

# Load data
df = pd.read_csv("", index_col=1)
D = df['subject_has_sarcopenia'].astype(int)
R = df['instability_rate']

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 4.5))

# ====================================================
# LEFT: ROC Curve
# ====================================================
fpr, tpr, thresholds = roc_curve(D, R)
roc_auc = auc(fpr, tpr)

ax1.plot(fpr, tpr, linewidth=1.8,
         label=f"AUC = {roc_auc:.3f}")
ax1.plot([0, 1], [0, 1], linestyle='--', linewidth=1)

# Operating point at threshold = 0.5
# idx = np.argmin(np.abs(thresholds - 0.5))
# ax1.plot(fpr[idx], tpr[idx], marker='o', markersize=5, )

ax1.set_xlabel("False Positive Rate")
ax1.set_ylabel("True Positive Rate")
ax1.set_title("ROC Curve")
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1])
ax1.legend(frameon=False)

# ====================================================
# RIGHT: Instability Rate Distribution
# ====================================================
bins = np.linspace(0, 1, 21)

ax2.hist(R[D == 0], bins=bins, alpha=0.6,
         label=f"Normal",
        linewidth=0.4)

ax2.hist(R[D == 1], bins=bins, alpha=0.6,
         label=f"Sarcopenia",
        linewidth=0.4)

ax2.set_xlabel("Instability Rate")
ax2.set_ylabel("Count")
ax2.set_title("Distribution by Sarcopenia Status")
ax2.legend(frameon=False)

plt.tight_layout()

with PdfPages("discrimination_analysis_combined.pdf") as pdf:
    pdf.savefig(fig, bbox_inches='tight', dpi=300)
    plt.close()

print("Saved cleaned IEEE-style figure.")
