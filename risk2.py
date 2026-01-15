import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Assume df is already loaded with columns: subject_has_sarcopenia, instability_rate

df = pd.read_csv("csvs/subjects_r.csv", index_col=1)
D = df['subject_has_sarcopenia'].astype(int)
R = df['instability_rate']

# Create figure
fig, ax = plt.subplots(figsize=(6, 4))

# Create histogram with overlapping distributions
bins = np.linspace(0, 1, 21)  # 20 bins from 0 to 1

# Plot for non-sarcopenia (controls)
ax.hist(R[D == 0], bins=bins, alpha=0.6, label='No sarcopenia (n=93)', 
        color='#2ecc71', edgecolor='black', linewidth=0.5)

# Plot for sarcopenia
ax.hist(R[D == 1], bins=bins, alpha=0.6, label='Sarcopenia (n=31)', 
        color='#e74c3c', edgecolor='black', linewidth=0.5)

# # Add vertical lines for key thresholds
# ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1.5, 
#            label='Classification threshold (γ=0.5)', alpha=0.7)

# Styling
ax.set_xlabel('Instability rate (R)')
ax.set_ylabel('Number of subjects')
# ax.set_title('Distribution of instability rates by sarcopenia status',  fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='upper center', framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

# Add text annotations for the peaks
peak_0_count_control = np.sum((R[D == 0] == 0))
peak_1_count_control = np.sum((R[D == 0] == 1))
peak_0_count_sarco = np.sum((R[D == 1] == 0))
peak_1_count_sarco = np.sum((R[D == 1] == 1))

# # Annotate the concentration at R=0 and R=1
# ax.text(0.05, ax.get_ylim()[1] * 0.95, 
#         f'R=0:\nControl: {peak_0_count_control}\nSarcopenia: {peak_0_count_sarco}',
#         fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
#         verticalalignment='top')

# ax.text(0.95, ax.get_ylim()[1] * 0.95, 
#         f'R=1:\nControl: {peak_1_count_control}\nSarcopenia: {peak_1_count_sarco}',
#         fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
#         verticalalignment='top', horizontalalignment='right')

# # Add summary statistics
# intermediate_count_control = np.sum((R[D == 0] > 0) & (R[D == 0] < 1))
# intermediate_count_sarco = np.sum((R[D == 1] > 0) & (R[D == 1] < 1))

# stats_text = f'Distribution Summary:\n'
# stats_text += f'Stable (R=0): {peak_0_count_control + peak_0_count_sarco} subjects (48.4%)\n'
# stats_text += f'Intermediate: {intermediate_count_control + intermediate_count_sarco} subjects (29.8%)\n'
# stats_text += f'Unstable (R=1): {peak_1_count_control + peak_1_count_sarco} subjects (21.8%)'

# ax.text(0.5, ax.get_ylim()[1] * 0.65, stats_text,
#         fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
#         verticalalignment='top', horizontalalignment='center')

plt.tight_layout()

# Save to PDF
with PdfPages('instability_rate_distribution.pdf') as pdf:
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

print("Figure saved to 'instability_rate_distribution.pdf'")
