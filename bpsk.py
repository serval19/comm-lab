import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

# ---------------------- Load Image from Fixed Location ----------------------
image_path = r"C:\Users\VICTUS\Desktop\cameraman.png"  # Update if needed
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"Error: Image not found at {image_path}")

rows, cols = image.shape

# ---------------------- Convert Image to Binary ----------------------
image_1d = image.flatten()  # Flatten image to 1D array
image_binary = [format(pixel, '08b') for pixel in image_1d]  # Convert pixels to 8-bit binary

# ---------------------- BPSK Modulation ----------------------
bpsk_symbols = np.array([1 if bit == '0' else -1 for binary in image_binary for bit in binary])

# Plot the constellation diagram of the transmitted BPSK signal
plt.figure(figsize=(6, 6))
plt.scatter(bpsk_symbols, np.zeros_like(bpsk_symbols), marker='o', color='b', s=1)
plt.xlabel("In-phase")
plt.ylabel("Quadrature")
plt.title("BPSK Constellation Diagram (Transmitted Signal)")
plt.grid()
plt.show()

# ---------------------- Define SNR Range and Initialize BER List ----------------------
snr_values = np.arange(-10, 21, 5)  # SNR values from -10 dB to 20 dB
ber_values = []

# ---------------------- Transmission Through AWGN Channel ----------------------
for snr_db in snr_values:
    snr_linear = 10 ** (snr_db / 10)  # Convert SNR from dB to linear scale
    noise_variance = 1 / (2 * snr_linear)  # Compute noise variance
    
    # Generate complex AWGN noise
    noise_real = np.random.normal(0, np.sqrt(noise_variance), size=bpsk_symbols.shape)
    noise_imag = np.random.normal(0, np.sqrt(noise_variance), size=bpsk_symbols.shape)
    noise = noise_real + 1j * noise_imag
    
    # Transmit through channel with AWGN
    received_signal = bpsk_symbols + noise
    
    # Plot the constellation diagram of the received signal
    plt.figure(figsize=(6, 6))
    plt.scatter(received_signal.real, received_signal.imag, marker='o', color='r', s=1)
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.title(f"BPSK Constellation Diagram (Received Signal, SNR={snr_db} dB)")
    plt.grid()
    plt.show()
    
    # ---------------------- BPSK Demodulation ----------------------
    decoded_symbols = np.where(received_signal.real >= 0, 1, -1)  # Decision threshold at 0
    decoded_bits = ['0' if symbol == 1 else '1' for symbol in decoded_symbols]  # Convert back to bits
    
    # Group bits into 8-bit binary values
    decoded_binary_values = ["".join(decoded_bits[i:i+8]) for i in range(0, len(decoded_bits), 8)]
    
    # Convert binary values back to pixel values
    decoded_pixels = [int(binary, 2) for binary in decoded_binary_values]

    
    # Reshape back to original image dimensions
    decoded_image = decoded_pixel_values.reshape(rows, cols)
    
    # Apply median filter for noise reduction
    denoised_image = median_filter(decoded_image, size=3)
    
    # Show the denoised reconstructed image at SNR = 10 dB
    if snr_db == 10:
        plt.figure(figsize=(6,6))
        plt.imshow(denoised_image, cmap='gray')
        plt.title(f"Denoised Reconstructed Image (SNR={snr_db} dB)")
        plt.axis("off")
        plt.show()
    
    # ---------------------- Compute BER ----------------------
    original_bits = [bit for binary in image_binary for bit in binary]
    num_bit_errors = sum(1 for o, d in zip(original_bits, decoded_bits) if o != d)
    bit_error_rate = num_bit_errors / len(original_bits)
    ber_values.append(bit_error_rate)
    
    print(f"SNR: {snr_db} dB, Bit Error Rate (BER): {bit_error_rate}")

# ---------------------- Plot BER vs SNR ----------------------
plt.figure(figsize=(8, 6))
plt.semilogy(snr_values, ber_values, marker='o', linestyle='-')
plt.xlabel("SNR (dB)")
plt.ylabel("Bit Error Rate (BER)")
plt.title("BER vs SNR for BPSK")
plt.grid(True, which='both')
plt.show()