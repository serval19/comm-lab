import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from scipy.special import erfc

# Read and preprocess image
filename = "D:/cameraman.png"  # Ensure the image file is in the same directory as the script
img = imageio.imread(filename)

# Convert to grayscale if the image is RGB
if len(img.shape) == 3:
    img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

# Crop to 256x256 for consistency
img = img[:256, :256]

# Display the original image
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.title("Original Image")
plt.show()

# Function to perform MPSK modulation
def modulate(bits, M):
    k = int(np.log2(M))
    bit_groups = bits.reshape(-1, k)
    symbols = []
    for b in bit_groups:
        decimal = 0
        for i in range(len(b)):
            decimal += b[i] * (2 ** (len(b) - 1 - i))
        symbols.append(decimal)
    symbols = np.array(symbols)
    angles = 2 * np.pi * symbols / M
    return np.cos(angles) + 1j * np.sin(angles)

# Function to add AWGN noise
def add_noise(signal, snr_db, bits_per_symbol):
    snr_linear = 10**(snr_db / 10)
    noise_std = np.sqrt(1 / (2 * bits_per_symbol * snr_linear))
    noise = noise_std * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
    return signal + noise

# Function to demodulate MPSK
def demodulate(received_signal, M):
    angles = np.angle(received_signal)
    decoded_symbols = np.round((angles / (2 * np.pi)) * M) % M
    return decoded_symbols.astype(int)

# Theoretical BER and SER formulas
def theoretical_ber(M, snr_db):
    k = np.log2(M)
    return erfc(np.sqrt(k * 10**(snr_db / 10)) / np.sqrt(2)) / k

def theoretical_ser(M, snr_db):
    return 2 * erfc(np.sqrt(2 * 10**(snr_db / 10)) * np.sin(np.pi / M))

# SNR range
snr_db_range = np.arange(-10, 11, 1)  # Adjust SNR steps for clearer results
M_values = [2, 4, 8]

BER_results = {}
SER_results = {}

for M in M_values:
    print(f"Processing for M={M}...")
    bits_per_symbol = int(np.log2(M))
    flattened_img = img.flatten()  # Convert 2D image to 1D array
    bit_stream = np.unpackbits(flattened_img) # Convert each value to 8 bits (0s and 1s
    extra_bits = len(bit_stream) % bits_per_symbol  # Find leftover bits
    if extra_bits != 0:
        bit_stream = bit_stream[:-extra_bits] 
      # Ensure correct length
    transmitted_symbols = modulate(bit_stream, M)
    
    BER_sim = []
    SER_sim = []
    
    for snr_db in snr_db_range:
        received_symbols = add_noise(transmitted_symbols, snr_db, bits_per_symbol)
        decoded_symbols = demodulate(received_symbols, M)
        decoded_bits = np.array([list(np.binary_repr(s, width=bits_per_symbol)) for s in decoded_symbols]).astype(int).flatten()
        
        bit_errors = np.sum(decoded_bits[:len(bit_stream)] != bit_stream)
        symbol_errors = np.sum(decoded_symbols[:len(transmitted_symbols)] != np.round(np.angle(transmitted_symbols) / (2 * np.pi) * M) % M)
        
        BER_sim.append(bit_errors / len(bit_stream))
        SER_sim.append(symbol_errors / len(transmitted_symbols))
        
        # Plot Constellation and Reconstructed Image for each SNR
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Constellation Diagram
        axes[0].scatter(received_symbols.real, received_symbols.imag, s=1)
        axes[0].set_xlabel('Real Part')
        axes[0].set_ylabel('Imaginary Part')
        axes[0].set_title(f'Received Signal Constellation (M={M}, SNR={snr_db} dB)')
        axes[0].grid(True)
        
        # Reconstructed Image
        reconstructed_bits = decoded_bits[:len(img.flatten()) * 8]  # Ensure correct number of bits
        reconstructed_image = np.packbits(reconstructed_bits)[:len(img.flatten())].reshape(img.shape)
        
        axes[1].imshow(reconstructed_image, cmap='gray')
        axes[1].set_title(f'Reconstructed Image (M={M}, SNR={snr_db} dB)')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    BER_results[M] = BER_sim
    SER_results[M] = SER_sim
    
    # Plot BER & SER for each M separately
    plt.figure(figsize=(8, 5))
    plt.semilogy(snr_db_range, BER_results[M], marker='o', linestyle='-', label=f'Practical BER M={M}')
    plt.semilogy(snr_db_range, [theoretical_ber(M, snr) for snr in snr_db_range], linestyle='--', label=f'Theoretical BER M={M}')
    plt.semilogy(snr_db_range, SER_results[M], marker='s', linestyle='-', label=f'Practical SER M={M}')
    plt.semilogy(snr_db_range, [theoretical_ser(M, snr) for snr in snr_db_range], linestyle='--', label=f'Theoretical SER M={M}')
    
    plt.xlabel('SNR (dB)')
    plt.ylabel('Error Rate')
    plt.title(f'BER & SER for M={M} over AWGN Channel')
    plt.grid(True, which='both')
    plt.legend()
    plt.show()

# Combined BER & SER plot for all M values
plt.figure(figsize=(10, 6))
for M in M_values:
    plt.semilogy(snr_db_range, BER_results[M], marker='o', linestyle='-', label=f'BER M={M}')
    plt.semilogy(snr_db_range, SER_results[M], marker='s', linestyle='--', label=f'SER M={M}')
    
plt.xlabel('SNR (dB)')
plt.ylabel('Error Rate')
plt.title('Combined BER and SER for MPSK over AWGN Channel')
plt.grid(True, which='both')
plt.legend()
plt.show()