import numpy as np
import matplotlib.pyplot as plt

# Reuse the Gaussian random variable X from before
np.random.seed(42)
r = 10
sigma2 = 1
n_samples = 10000
X = np.random.normal(r, np.sqrt(sigma2), n_samples)

# PCM on X for N = 2 to 6
N_values = np.arange(2, 7)
SQNR = []

for N in N_values:
    L = 2 ** N
    delta = (max(X) - min(X)) / L
    q_levels = np.round((X - min(X)) / delta) * delta + min(X)
    noise = X - q_levels
    sqnr = 10 * np.log10(np.var(X) / np.var(noise))
    SQNR.append(sqnr)

# Plot SQNR vs. N
plt.figure(figsize=(8, 5))
plt.plot(N_values, SQNR, 'ro-', label='SQNR vs. N (Gaussian Signal)')
plt.xlabel('N (Quantization Bits)')
plt.ylabel('SQNR (dB)')
plt.title('SQNR vs. Number of Bits for Gaussian Signal')
plt.grid(True)
plt.legend()
plt.show()

# Display PCM Encoded Output for N = 4
N = 4
L = 2 ** N
delta = (max(X) - min(X)) / L
q_levels = np.round((X - min(X)) / delta) * delta + min(X)
normalized = (q_levels - min(X)) / delta
int_vals = normalized.astype(int)
pcm_encoded_output = [format(val, f'0{N}b') for val in int_vals[:10]]  # Show first 10 encoded values

print(f"PCM Binary Encoding for N = {N}:")
print(pcm_encoded_output)
