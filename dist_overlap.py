import numpy as np
import matplotlib.pyplot as plt

# Parameters
mu1,sigma1 = 0.16355327976321726, 0.2932682824670479
mu2,sigma2 = 0.2956865040966423, 0.38241392820985254

# x range
x = np.linspace(min(mu1-4*sigma1, mu2-4*sigma2), max(mu1+4*sigma1, mu2+4*sigma2), 1000)

# Normal PDFs
y1 = (1/(sigma1*np.sqrt(2*np.pi))) * np.exp(-(x-mu1)**2 / (2*sigma1**2))
y2 = (1/(sigma2*np.sqrt(2*np.pi))) * np.exp(-(x-mu2)**2 / (2*sigma2**2))

# Plot distributions
plt.plot(x, y1, color='blue', label='Normal 1')
plt.plot(x, y2, color='red', label='Normal 2')

# Highlight overlap
plt.fill_between(x, y1, y2, where=(y2<y1), color='blue', alpha=0.5)
plt.fill_between(x, y1, y2, where=(y1<=y2), color='red', alpha=0.5)

plt.legend()
plt.show()
