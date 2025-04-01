import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import convolve

def raised_cosine_filter(beta, L, Nsym):
    """Generates a Raised Cosine Filter"""
    t = np.arange(-Nsym / 2, Nsym / 2 + 1 / L, 1 / L)
    sinc_part = np.sinc(t)
    cos_part = np.cos(np.pi * beta * t)
    denom = 1 - (2 * beta * t) ** 2
    p = sinc_part * cos_part
    p[denom != 0] /= denom[denom != 0]  # Avoid division by zero
    return p / np.sqrt(np.sum(p ** 2))  # Normalize energy

# Read the image
image = cv2.imread('D:/cameraman.png', cv2.IMREAD_GRAYSCALE)
assert image is not None, "Image not found!"

# Convert pixel values to bits and map to BPSK symbols
bits = np.unpackbits(image)
bpsk_symbols = 2 * bits - 1  # Map 0 -> -1, 1 -> 1

# Parameters
L = 4  # Upsampling factor
Tsym = 1  # Symbol duration
Nsym = 8  # Filter length in symbols
SNR_values = [-10, -5, 5, 10]
beta_values = [0.2, 0.8]

fig, axes = plt.subplots(len(SNR_values), len(beta_values) * 2, figsize=(24, 12))

for i, SNR in enumerate(SNR_values):
    for j, beta in enumerate(beta_values):
        print(f"Processing SNR={SNR} dB, Beta={beta}")
        
        # Generate Raised Cosine filter
        p = raised_cosine_filter(beta, L, Nsym)
        
        # Upsampling
        upsampled_symbols = np.zeros(L * len(bpsk_symbols))
        upsampled_symbols[::L] = bpsk_symbols
        
        # Pulse shaping
        shaped_signal = convolve(upsampled_symbols, p, mode='same')
        
        # Add Gaussian noise
        noise_power = 1 / (2 * (10 ** (SNR / 10)))
        noise = np.sqrt(noise_power) * (np.random.randn(len(shaped_signal)) + 1j * np.random.randn(len(shaped_signal)))
        received_signal = shaped_signal + noise
        
        # Matched filtering
        g = p[::-1]  # Matched filter
        matched_output = convolve(received_signal, g, mode='same')
        
        # Downsampling and demapping
        downsampled_output = matched_output[::L]
        demapped_bits = (downsampled_output.real > 0).astype(np.uint8)
        
        # Convert bits to pixel values
        received_image = np.packbits(demapped_bits).reshape(image.shape)
        
        # Plot Reconstructed Image
        axes[i, j * 2].imshow(received_image, cmap='gray')
        axes[i, j * 2].set_title(f"Reconstructed Image (SNR={SNR} dB, Beta={beta})", fontsize=10)
        axes[i, j * 2].axis('off')
        
        # Plot Eye Diagram
        nSamples = 3 * L
        nTraces = 100
        samples = downsampled_output[:nSamples * nTraces].reshape(nTraces, nSamples)
        for trace in samples:
            axes[i, j * 2 + 1].plot(trace, color='orange', alpha=0.7)
        axes[i, j * 2 + 1].set_title(f'Eye Diagram (SNR={SNR} dB, Beta={beta})', fontsize=10)
        axes[i, j * 2 + 1].grid(True)
        axes[i, j * 2 + 1].set_xticks([])
        axes[i, j * 2 + 1].set_yticks([])

plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.show()