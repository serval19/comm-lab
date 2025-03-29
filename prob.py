import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Set the random seed for reproducibility
np.random.seed(42)

# Roll number (mean r)
r = 10

# Variance σ^2
sigma2 = 1

# Number of samples
n_samples = 10000

# Generate random variables X and Y
X = np.random.normal(r, np.sqrt(sigma2), n_samples)
Y = np.random.normal(r, np.sqrt(sigma2), n_samples)



# Plot histogram for X
plt.subplot(1, 2, 1)
plt.hist(X, bins=30, density=True, alpha=0.6, color='g', label='X samples')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, r, np.sqrt(sigma2))
plt.plot(x, p, 'k', linewidth=2, label='Gaussian PDF')
plt.title('Histogram and PDF of X')
plt.xlabel('X')
plt.ylabel('Density')
plt.legend()

# Plot histogram for Y
plt.subplot(1, 2, 2)
plt.hist(Y, bins=30, density=True, alpha=0.6, color='b', label='Y samples')
xmin, xmax = plt.xlim()
p = norm.pdf(x, r, np.sqrt(sigma2))
plt.plot(x, p, 'k', linewidth=2, label='Gaussian PDF')
plt.title('Histogram and PDF of Y')
plt.xlabel('Y')
plt.ylabel('Density')
plt.legend()
plt.show()


# Step (c): Generate a new random variable Z = X + Y
Z = X + Y

# Plot the histogram for Z
plt.hist(Z, bins=30, density=True, alpha=0.6, color='purple', label='Z samples')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, 2 * r, np.sqrt(2))  # Mean of Z is 2r, variance of Z is 2
plt.plot(x, p, 'k', linewidth=2, label='Gaussian PDF for Z')
plt.title('Histogram and PDF of Z = X + Y')
plt.xlabel('Z')
plt.ylabel('Density')
plt.legend()
plt.show()

# Step (c): Calculate the mean and variance of Z
mean_Z = np.mean(Z)
var_Z = np.var(Z)
print(f'Mean of Z: {mean_Z}')
print(f'Variance of Z: {var_Z}')