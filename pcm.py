import numpy as np
import matplotlib.pyplot as plt
# Pulse Coded Modulation Parameters
fs = 4 * 8 # Sampling at 4 times Nyquist Rate
t = np.linspace(0, 1, fs)
mod_r = STUDENT_ROLL_NO % 5 + 1 # Calculate mod(r,5)+1
s_t = mod_r * (1 + np.cos(8 * np.pi * t)) / 2
# Quantization Levels L
L_values = [4, 8, 16, 32, 64]
SQNR = []
for L in L_values:
delta = (max(s_t) - min(s_t)) / L
q_levels = np.round((s_t - min(s_t)) / delta) * delta + min(s_t)
noise = s_t - q_levels
sqnr = 10 * np.log10(np.var(s_t) / np.var(noise))
SQNR.append(sqnr)
# a) Plot SQNR vs. N
N_values = np.log2(L_values)
plt.figure(figsize=(10, 5))
plt.plot(N_values, SQNR, 'bo-', label='SQNR vs. N')
plt.xlabel('N = log2(L)')
plt.ylabel('SQNR (dB)')
plt.title('SQNR vs. Quantization Bits')
plt.legend()
plt.show()
# b) PCM Modulated Output for L=32
L = 32
delta = (max(s_t) - min(s_t)) / L
q_levels = np.round((s_t - min(s_t)) / delta) * delta + min(s_t)
# Step 1: Calculate the normalized values
normalized_values = (q_levels - min(s_t)) / delta

# Step 2: Convert the normalized values to integers
integer_values = normalized_values.astype(int)

# Step 3: Convert each integer to a 5-digit binary string
binary_encoded = []
for i in integer_values:
    binary_string = format(i, '05b')  # Convert to 5-digit binary
    binary_encoded.append(binary_string)

# Now, binary_encoded contains the binary strings
print("PCM Binary Encoding for L=32:")
print(binary_encoded[:10]) # Print first 10 encoded values
