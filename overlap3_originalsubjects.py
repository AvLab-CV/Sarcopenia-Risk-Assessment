import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_auc_score, roc_curve

# FOLD = 1
# WINDOW_SIZE = 64
# STRIDE = 8
# SKIP = 1 # <- window stride. 1 = normal sliding window behavior
# df = pd.read_csv(f"./output/results/fold{FOLD}_sliding_window_windowsize{WINDOW_SIZE}_stride{STRIDE}.csv")
# df['predictions'] = df['predictions'].map(lambda x: np.fromstring(x.strip("[ ]"), sep=' '))
# # Stride 8 -> 8*SKIP
# df['predictions'] = df['predictions'].map(lambda x: x[::SKIP])
# largest_window = df['predictions'].map(lambda x: x.shape[0]).max()
# # window_pos[i] = position at the beginning window i
# window_pos = np.arange(largest_window) * STRIDE
# df['window_pos'] = df['predictions'].map(lambda x: window_pos[:x.shape[0]])
# df['stable_count']   = df['predictions'].map(lambda x: (x <= 0.5).sum())
# df['unstable_count'] = df['predictions'].map(lambda x: (x  > 0.5).sum())

df = pd.read_csv("csvs/subjects.csv", index_col=0)
print(df)
df['subject_has_sarcopenia'] = df['sarcopenia-normal'] == 'sarcopenia'
df['unstable_count'] = df['unstable']
df['stable_count'] = df['stable']
df['instability_rate'] = df['unstable_count'] / (df['stable_count'] + df['unstable_count'])
df.to_csv("csvs/subjects_r.csv")

# Your existing data
normal_data = df.loc[df['subject_has_sarcopenia'] == 0, 'instability_rate'].values
sarco_data = df.loc[df['subject_has_sarcopenia'] == 1, 'instability_rate'].values

# Prior probabilities (from your dataset)
n_normal = len(normal_data)
n_sarco = len(sarco_data)
P_sarco = n_sarco / (n_normal + n_sarco)  # ~0.21 based on your 31/142
P_normal = 1 - P_sarco

# Estimate conditional densities with KDE (with boundary correction)
def fit_kde_bounded(data):
    reflected = np.concatenate([data, -data, 2 - data])
    return gaussian_kde(reflected, bw_method='scott')

kde_sarco = fit_kde_bounded(sarco_data)
kde_normal = fit_kde_bounded(normal_data)

# Compute posterior probability P(sarcopenia | R = r)
r_values = np.linspace(0, 1, 200)

p_r_given_sarco = kde_sarco(r_values)
p_r_given_normal = kde_normal(r_values)

# Marginal density p(r)
p_r = P_sarco * p_r_given_sarco + P_normal * p_r_given_normal

# Posterior probability (Bayes' theorem)
P_sarco_given_r = (p_r_given_sarco * P_sarco) / p_r

# Create comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# fig.suptitle(f"Partition {1} Windowsize={WINDOW_SIZE} Stride={STRIDE*SKIP}")

# 1. Conditional densities
ax = axes[0, 0]
# add samples
colors = df['subject_has_sarcopenia'].map({0: 'blue', 1: 'orange'})
ax.scatter(df['instability_rate'], 
           np.zeros_like(df['instability_rate']) - 0.002, 
           c=colors, marker='x', s=25, alpha=0.7)
# plot conditional densities
ax.plot(r_values, p_r_given_normal, 'blue', label='p(r|normal)', lw=2)
ax.plot(r_values, p_r_given_sarco, 'orange', label='p(r|sarcopenia)', lw=2)
ax.fill_between(r_values, 0, np.minimum(p_r_given_normal, p_r_given_sarco), 
                alpha=0.3, color='gray', label='Overlap')
ax.set_xlabel('Instability rate (r)')
ax.set_ylabel('Density')
# ax.set_xlim(0.0, 1.0)
# ax.set_ylim(0.0, 1.0)
ax.set_title('Conditional distributions (KDE)')
ax.legend()

# Calculate overlap coefficient
overlap = np.trapz(np.minimum(p_r_given_normal, p_r_given_sarco), r_values)
ax.text(0.5, 0.95, f'Overlap coefficient: {overlap:.3f}', 
        transform=ax.transAxes, va='top', bbox=dict(boxstyle='round', facecolor='wheat'))

# 2. Posterior probability P(sarcopenia | r)
ax = axes[0, 1]
ax.plot(r_values, P_sarco_given_r, 'red', lw=2)
ax.axhline(P_sarco, color='black', linestyle='--', label=f'Prior P(sarco) = {P_sarco:.3f}')
ax.set_xlabel('Instability rate (r)')
ax.set_ylabel('P(sarcopenia | r)')
# ax.set_xlim(0.0, 1.0)
# ax.set_ylim(0.0, 1.0)
ax.set_title('Posterior Probability of Sarcopenia')
ax.legend()
ax.grid(True, alpha=0.3)

# Add reference lines
# ax.axhline(0.5, color='green', linestyle=':', alpha=0.5, label='50% threshold')

# 3. ROC Curve
ax = axes[1, 0]
y_true = df['subject_has_sarcopenia'].astype(int).values
y_score = df['instability_rate'].values
fpr, tpr, thresholds = roc_curve(y_true, y_score)
auc = roc_auc_score(y_true, y_score)

ax.plot(fpr, tpr, 'b-', lw=2, label=f'ROC (AUC = {auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', label='Random classifier')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Likelihood Ratio
ax = axes[1, 1]
# Avoid division by zero
epsilon = 1e-10
likelihood_ratio = p_r_given_sarco / (p_r_given_normal + epsilon)
ax.plot(r_values, likelihood_ratio, 'purple', lw=2)
ax.axhline(1, color='black', linestyle='--', label='LR = 1 (no information)')
ax.set_xlabel('Instability rate (r)')
ax.set_ylabel('Likelihood Ratio')
ax.set_title('p(r|sarcopenia) / p(r|normal)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

plt.tight_layout()
plt.savefig('bayesian_analysis.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Print key statistics
print("=" * 60)
print("DISTRIBUTION SEPARATION METRICS")
print("=" * 60)

# Effect size
pooled_std = np.sqrt(((n_normal-1)*normal_data.std()**2 + 
                       (n_sarco-1)*sarco_data.std()**2) / 
                      (n_normal + n_sarco - 2))
cohens_d = (sarco_data.mean() - normal_data.mean()) / pooled_std
print(f"Cohen's d: {cohens_d:.3f}")

# AUC
print(f"AUC-ROC: {auc:.3f}")

# Overlap
print(f"Overlap coefficient: {overlap:.3f}")

# Mann-Whitney U test
from scipy.stats import mannwhitneyu
u_stat, p_value = mannwhitneyu(sarco_data, normal_data, alternative='greater')
print(f"Mann-Whitney U test: p = {p_value:.4f}")

# Risk at different instability rates
print("\n" + "=" * 60)
print("RISK AT DIFFERENT INSTABILITY RATES")
print("=" * 60)
for r_threshold in [0.0, 0.2, 0.4, 0.6, 0.8]:
    idx = np.argmin(np.abs(r_values - r_threshold))
    risk = P_sarco_given_r[idx]
    print(f"P(sarcopenia | r = {r_threshold:.1f}) = {risk:.3f} ({risk/P_sarco:.2f}x baseline risk)")

# Monotonicity check
correlation = np.corrcoef(r_values, P_sarco_given_r)[0, 1]
print(f"\nCorrelation between r and P(sarco|r): {correlation:.4f}")
if correlation > 0.99:
    print("✓ Hypothesis II confirmed: Monotonically increasing relationship")
else:
    print("⚠ Warning: Non-monotonic relationship detected")
