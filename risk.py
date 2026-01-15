import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from scipy import stats
import matplotlib.pyplot as plt

# Assume df is already loaded
df = pd.read_csv("csvs/subjects_r.csv", index_col=1)
D = df['subject_has_sarcopenia'].astype(int)
R = df['instability_rate']

# ============================================
# 1. CALIBRATION
# ============================================

# Calibration plot - group by deciles of R
df['R_decile'] = pd.qcut(R, q=10, labels=False, duplicates='drop')
calibration = df.groupby('R_decile').agg({
    'instability_rate': 'mean',  # Mean predicted risk
    'subject_has_sarcopenia': 'mean'  # Observed rate
}).reset_index()

# Calibration slope and intercept
from sklearn.linear_model import LogisticRegression
# Fit logistic regression: logit(P(D=1)) = intercept + slope * R
# Ideal: intercept=0, slope=1
cal_model = LogisticRegression()
cal_model.fit(R.values.reshape(-1, 1), D)
cal_slope = cal_model.coef_[0][0]
cal_intercept = cal_model.intercept_[0]

print("=== CALIBRATION ===")
print(f"Calibration slope: {cal_slope:.3f} (ideal: 1.0)")
print(f"Calibration intercept: {cal_intercept:.3f} (ideal: 0.0)")
print("\nCalibration by decile:")
print(calibration)

# Plot calibration
plt.figure(figsize=(8, 6))
plt.scatter(calibration['instability_rate'], calibration['subject_has_sarcopenia'])
plt.plot([0, 1], [0, 1], 'r--', label='Perfect calibration')
plt.xlabel('Mean Predicted Risk (R)')
plt.ylabel('Observed Disease Rate')
plt.title('Calibration Plot')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ============================================
# 2. CLINICAL CONTEXT
# ============================================

baseline_prevalence = D.mean()
print(f"\n=== CLINICAL CONTEXT ===")
print(f"Baseline disease prevalence: {baseline_prevalence:.1%}")
print(f"Sample size: {len(df)}")
print(f"Cases: {D.sum()}, Controls: {(~D.astype(bool)).sum()}")

# ============================================
# 3. COMPARISON (to baseline)
# ============================================

auc = roc_auc_score(D, R)
print(f"\n=== COMPARISON ===")
print(f"AUC of R: {auc:.3f}")
print(f"Improvement over random: {(auc - 0.5):.3f}")

# ============================================
# 4. RISK GRADIENT
# ============================================

print(R.describe())
print(f"\nValue counts:\n{R.value_counts().sort_index()}")
print(f"\nUnique values: {R.nunique()}")

# Quintile analysis
# Quintile analysis - handle duplicates
df['R_quintile'] = pd.qcut(R, q=5, duplicates='drop')
# Relabel based on actual number of quantiles created
unique_quantiles = df['R_quintile'].nunique()
df['R_quintile'] = df['R_quintile'].cat.rename_categories([f'Q{i+1}' for i in range(unique_quantiles)])
# df['R_quintile'] = pd.qcut(R.rank(method='first'), q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
risk_gradient = df.groupby('R_quintile').agg({
    'subject_has_sarcopenia': ['sum', 'count', 'mean'],
    'instability_rate': ['mean', 'min', 'max']
}).round(3)

print("\n=== REVISED RISK GRADIENT ===")

# Create meaningful risk groups based on actual distribution
def categorize_risk(r):
    if r == 0.0:
        return 'Stable (R=0)'
    elif r == 1.0:
        return 'Unstable (R=1)'
    else:
        return 'Intermediate (0<R<1)'

df['risk_group'] = df['instability_rate'].apply(categorize_risk)

risk_summary = df.groupby('risk_group').agg({
    'subject_has_sarcopenia': ['sum', 'count', 'mean'],
    'instability_rate': 'mean'
}).round(3)

print(risk_summary)

# Calculate odds ratios
stable_rate = df[df['risk_group'] == 'Stable (R=0)']['subject_has_sarcopenia'].mean()
intermediate_rate = df[df['risk_group'] == 'Intermediate (0<R<1)']['subject_has_sarcopenia'].mean()
unstable_rate = df[df['risk_group'] == 'Unstable (R=1)']['subject_has_sarcopenia'].mean()

print(f"\nDisease rates by group:")
print(f"  Stable (R=0): {stable_rate:.1%}")
print(f"  Intermediate: {intermediate_rate:.1%}")
print(f"  Unstable (R=1): {unstable_rate:.1%}")

# Odds ratios
if stable_rate > 0 and stable_rate < 1:
    stable_odds = stable_rate / (1 - stable_rate)
    unstable_odds = unstable_rate / (1 - unstable_rate)
    or_unstable_vs_stable = unstable_odds / stable_odds
    print(f"\nOdds Ratio (Unstable vs Stable): {or_unstable_vs_stable:.2f}")
    
    if intermediate_rate > 0 and intermediate_rate < 1:
        intermediate_odds = intermediate_rate / (1 - intermediate_rate)
        or_intermediate_vs_stable = intermediate_odds / stable_odds
        print(f"Odds Ratio (Intermediate vs Stable): {or_intermediate_vs_stable:.2f}")

# Statistical test across groups
from scipy.stats import chi2_contingency
contingency = pd.crosstab(df['risk_group'], df['subject_has_sarcopenia'])
chi2, p_value, dof, expected = chi2_contingency(contingency)
print(f"\nChi-square test: p = {p_value:.4f}")


# ============================================
# 5. ROC CURVE
# ============================================

fpr, tpr, thresholds = roc_curve(D, R)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'R (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ============================================
# 6. DISTRIBUTION BY DISEASE STATUS
# ============================================

plt.figure(figsize=(10, 6))
plt.hist(R[D == 0], bins=20, alpha=0.5, label='No Sarcopenia', density=True)
plt.hist(R[D == 1], bins=20, alpha=0.5, label='Sarcopenia', density=True)
plt.xlabel('Risk Score (R = instability_rate)')
plt.ylabel('Density')
plt.title('Distribution of Risk Score by Disease Status')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\n=== SUMMARY ===")
print(f"Risk score R (instability_rate) shows modest discrimination (AUC={auc:.3f})")
print(f"for sarcopenia, which has a baseline prevalence of {baseline_prevalence:.1%}.")
