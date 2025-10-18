import numpy as np
import struct

# Load weights and biases
w0 = np.loadtxt('layer_0_weights.csv', delimiter=',').astype(np.float32)
b0 = np.loadtxt('layer_0_bias.csv', delimiter=',').astype(np.float32)
w1 = np.loadtxt('layer_1_weights.csv', delimiter=',').astype(np.float32)
b1 = np.loadtxt('layer_1_bias.csv', delimiter=',').astype(np.float32)
w2 = np.loadtxt('layer_2_weights.csv', delimiter=',').astype(np.float32)
b2 = np.loadtxt('layer_2_bias.csv', delimiter=',').astype(np.float32)

def write_array(f, array):
    flat = array.flatten()
    length = len(flat)
    f.write(struct.pack('I', length))         # write length prefix
    f.write(struct.pack(f'{length}f', *flat)) # write float values

with open('quantized_weights.bin', 'wb') as f:
    write_array(f, w0)
    write_array(f, b0)
    write_array(f, w1)
    write_array(f, b1)
    write_array(f, w2)
    write_array(f, b2)

print(f"[INFO] Exported binary weights: quantized_weights.bin")
print(f"[INFO] Sizes:")
print(f" - w0: {w0.shape}, b0: {b0.shape}")
print(f" - w1: {w1.shape}, b1: {b1.shape}")
print(f" - w2: {w2.shape}, b2: {b2.shape}")