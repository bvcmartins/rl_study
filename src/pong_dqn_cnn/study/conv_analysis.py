import torch
import torch.nn as nn

# Let's trace through the conv layers step by step
input_shape = (1, 1, 84, 84)  # (batch_size, channels, height, width)
print(f"Input shape: {input_shape}")

# First conv layer: Conv2d(1, 32, kernel_size=8, stride=4)
# Output size formula: (input_size - kernel_size) / stride + 1
conv1_h = (84 - 8) // 4 + 1
conv1_w = (84 - 8) // 4 + 1
conv1_shape = (1, 32, conv1_h, conv1_w)
print(f"After Conv1 (k=8, s=4): {conv1_shape}")

# Second conv layer: Conv2d(32, 64, kernel_size=4, stride=2)
conv2_h = (conv1_h - 4) // 2 + 1
conv2_w = (conv1_w - 4) // 2 + 1
conv2_shape = (1, 64, conv2_h, conv2_w)
print(f"After Conv2 (k=4, s=2): {conv2_shape}")

# Third conv layer: Conv2d(64, 64, kernel_size=3, stride=1)
conv3_h = (conv2_h - 3) // 1 + 1
conv3_w = (conv2_w - 3) // 1 + 1
conv3_shape = (1, 64, conv3_h, conv3_w)
print(f"After Conv3 (k=3, s=1): {conv3_shape}")

# After flatten
flattened_size = 64 * conv3_h * conv3_w
print(f"Flattened size: {flattened_size}")

print("\n" + "="*50)
print("VERIFICATION WITH ACTUAL PYTORCH MODEL:")

# Now let's verify with actual PyTorch
conv_layers = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=8, stride=4),
    nn.ReLU(),
    nn.Conv2d(32, 64, kernel_size=4, stride=2),
    nn.ReLU(),
    nn.Conv2d(64, 64, kernel_size=3, stride=1),
    nn.ReLU(),
    nn.Flatten()
)

# Test with 84x84 input
test_input = torch.randn(1, 1, 84, 84)
with torch.no_grad():
    output = conv_layers(test_input)
    print(f"Actual output shape: {output.shape}")
    print(f"Actual flattened size: {output.shape[1]}")

print("\n" + "="*50)
print("WHY THIS ARCHITECTURE?")
print("1. 84x84 is standard for Atari games (from DeepMind's DQN paper)")
print("2. Large stride (4) in first layer reduces computation while preserving important features")
print("3. Progressively smaller kernels extract finer details")
print("4. Final 7x7 feature maps are small enough for efficient fully connected layers")