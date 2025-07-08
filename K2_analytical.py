import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
# Analytically calculates the 
# Parameters
sigma = 1.0  # Variance of the normal distribution

# Define the density function
# For x < 0: density = 0
# For x = 0: density = 1/2 (as a Dirac mass, shown as a bar)
# For x > 0: density = normal density with mean 0, variance sigma

def custom_density(x, sigma):
    density = np.zeros_like(x)
    mask = x > 0
    density[mask] = norm.pdf(x[mask], loc=0, scale=np.sqrt(sigma))
    return density

x = np.linspace(0, 5, 500)

y = custom_density(x, sigma)

plt.figure(figsize=(8, 4))
plt.plot(x, y)
plt.axvline(0, color='k', linestyle='--', alpha=0.5)
plt.bar(0, 0.5, width=0.05, color='red')
plt.xlabel("x")
plt.xlim(0, 5)
plt.ylabel("density")
plt.tight_layout()
plt.show()

