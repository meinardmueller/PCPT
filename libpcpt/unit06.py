"""
Module: libpcpt.unit06
Authors: Meinard Mueller, Johannes Zeitler, Sebastian Strahl, International Audio Laboratories Erlangen
License: The MIT license, https://opensource.org/licenses/MIT
This file is part of the PCPT Notebooks (https://www.audiolabs-erlangen.de/PCPT)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# ========= Visualizations =========
def plot_signal_denoise(t, x_input, y_target=None, y_output=None):
    """
    Plot an input signal along with optional target and predicted output.

    Args:
        t (ndarray): Time vector.
        x_input (ndarray): Input signal (e.g., noisy or mixed).
        y_target (ndarray, optional): Ground truth target signal.
        y_output (ndarray, optional): Output from the learned model or filter.
    """
    plt.figure(figsize=(4.5, 1.8))
    plt.plot(t, x_input, color='gray', linewidth=0.5, label='Input')

    if y_target is not None:
        plt.plot(t, y_target, color='black', linewidth=1.2, label='Target')

    if y_output is not None:
        plt.plot(t, y_output, color='red', linewidth=1.2, label='Output')

    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend(loc='upper right', fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8) 
    plt.tight_layout()
    plt.show()
    
def visualize_pairs_24(t, X, Y):
    """Visualization of first 24 noisy-clean pairs

    Notebook: PCPT_06_convolution.ipynb

    Args:
        t (np.ndarray): Shared time axis of shape (len_signal,)
        X (np.ndarray or Tensor): Noisy signals, shape (N, len_signal) or (N, 1, len_signal)
        Y (np.ndarray or Tensor): Clean signals, shape (N, len_signal) or (N, 1, len_signal)
    """
    n_rows, n_cols = 4, 6
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6, 2.8), sharex=True, sharey=True)
    axs = axs.flatten()
    for n in range(n_rows * n_cols):
        x = X[n]
        y = Y[n]
        axs[n].plot(t, x, color='gray', linewidth=0.5)
        axs[n].plot(t, y, color='black', linewidth=1.0)
        axs[n].axis('off')
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.suptitle('First 24 Training Pairs', fontsize=9)

def plot_filter_kernel(kernel_weights):
    plt.figure(figsize=(4.5, 1.8))
    plt.stem(kernel_weights, linefmt='k-', markerfmt='ko', basefmt=' ')
    plt.xticks(range(len(kernel_weights)))
    plt.xlabel("Tap index", fontsize=8)
    plt.ylabel("Weight", fontsize=8)
    plt.title("Learned Filter Kernel", fontsize=9)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)     
    plt.tight_layout()
    plt.show()

def visualize_further_examples_6(t, X, Y, Y_pred):
    """Visualize six further example

    Notebook: PCPT_06_convolution.ipynb

    Args:
        t (np.ndarray): Shared time vector of shape (len_signal,)
        X (np.ndarray): Noisy input signals, shape (N, len_signal)
        Y (np.ndarray): Clean target signals, shape (N, len_signal)
        Y_pred (np.ndarray): Model-predicted signals, shape (N, len_signal)
    """
    n_rows, n_cols = 2, 3
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6, 2.8), sharex=True, sharey=True)
    axs = axs.flatten()

    for n in range(n_rows * n_cols):
        axs[n].plot(t, X[n], color='gray', linewidth=0.5, label='Noisy input')
        axs[n].plot(t, Y[n], color='black', linewidth=1.0, label='Clean target')
        axs[n].plot(t, Y_pred[n], color='red', linewidth=1.0, label='Predicted output')
        axs[n].tick_params(labelsize=8)  # Reduce tick label font size
    plt.suptitle("Six Further Examples", fontsize=9)
    plt.subplots_adjust(hspace=0.2, wspace=0.1, top=0.88)
    plt.show()  
        
# ================================================
# Exercise 1: exercise_convolution_smooth_edge()
# ================================================
def exercise_convolution_smooth_edge():
    """Exercise 1: Convolution for Smoothing and Edge Detection

    Notebook: PCPT_06_convolution.ipynb
    """
    # Define signal
    x = np.array([1, 1, 0, 1, 1, 0, 8, 9, 7, 7, 6, 5, 4])

    # Define filters
    h_avg = np.array([1/3, 1/3, 1/3])     # Low-pass / smoothing filter
    h_edge = np.array([1, 0, -1])         # High-pass / edge detection filter

    # Convert into tensors and add proper number of dimensions
    x_pt = torch.tensor(x, dtype=torch.float32).view(1, 1, -1)
    h_avg_pt = torch.tensor(h_avg, dtype=torch.float32).view(1, 1, -1)
    h_edge_pt = torch.tensor(h_edge, dtype=torch.float32).view(1, 1, -1)

    # Manually flip kernel to perform convolution instead of cross-correlation
    x_smooth = F.conv1d(x_pt, h_avg_pt.flip(-1), padding="same").squeeze()   
    x_edges = F.conv1d(x_pt, h_edge_pt.flip(-1), padding="same").squeeze()

    # Sample indices
    t = np.arange(len(x))

    # Plot results
    plt.figure(figsize=(4.5, 2.3))
    plt.plot(t, x, 'k-', linewidth=1.5, label='Original')
    plt.plot(t, x, 'ko', markersize=3)

    plt.plot(t, x_smooth, 'c-', linewidth=1, label='Smoothing')
    plt.plot(t, x_smooth, 'co', markersize=2)

    plt.plot(t, x_edges, 'r-', linewidth=1, label='Edge detection')
    plt.plot(t, x_edges, 'ro', markersize=2)

    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.yticks([-4, -2, 0, 2, 4, 6, 8])  # Set custom ticks for signal values
    plt.legend(fontsize=8)
    plt.title("Convolution: Smoothing and Edge Detection", fontsize=9)
    plt.xlabel("Time index", fontsize=8)
    plt.ylabel("Signal value", fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)    
    plt.tight_layout()
    plt.show()

    print("* Smoothing reduces sharp transitions by averaging neighbors.")
    print("* The edge detector reacts to abrupt changes")
    print("  (e.g., when the signal rises from 0 to 8).")
    print("* Zero-padding can cause edge boundary artifacts")
    print(" (e.g., spurious negative values at the end).")
    
# ================================================
# Exercise 2: exercise_conv1d_parameters()
# ================================================
def exercise_conv1d_parameters():
    """Exercise 2: Understanding Key Parameters of nn.Conv1d

    Notebook: PCPT_06_convolution.ipynb
    """
    # Create a random batch of input signals: shape (B=3, C=2, T=15)
    B, C, T = 3, 2, 15  # Batch size, Channels, Time steps
    x = torch.randn(B, C, T)
    print(f"Input shape: {x.shape}\n")

    # Define convolution layers
    layers = [
        nn.Conv1d(in_channels=C, out_channels=3, kernel_size=5, stride=1, padding=0, bias=False),
        nn.Conv1d(in_channels=C, out_channels=3, kernel_size=5, stride=1, padding=0),
        nn.Conv1d(in_channels=C, out_channels=3, kernel_size=5, stride=3, padding=2),
        nn.Conv1d(in_channels=C, out_channels=7, kernel_size=5, stride=1, padding=2, dilation=3),
    ]

    # Analyze each layer
    for idx, conv in enumerate(layers, 1):
        y = conv(x)
        num_weights = conv.weight.numel()
        num_biases = conv.bias.numel() if conv.bias is not None else 0
        print(f"Layer {idx}: {conv}")
        print(f"  Output shape: {y.shape}")
        print(f"  Weights shape: {conv.weight.shape} -> total: {num_weights}")
        print(f"  Bias shape: {None if conv.bias is None else conv.bias.shape} -> total: {num_biases}")
        print(f"  Total parameters: {num_weights + num_biases}\n")

    # Case: stride > kernel_size
    conv_stride_kernel = nn.Conv1d(in_channels=C, out_channels=3, kernel_size=3, stride=5)
    y_stride_kernel = conv_stride_kernel(x)
    print("Case: stride > kernel_size:")
    print(f"Layer 5: {conv_stride_kernel}")
    print(f"  Output shape: {y_stride_kernel.shape}\n")
    print("Note: When stride > kernel_size, the convolution kernel spans a small window\n"
          "(e.g., 3 values) but advances by a larger step (e.g., 5).")
    print("As a result, parts of the input are skipped entirely and do not influence\n"
          "the output. This can lead to a loss of fine signal details, reducing\n"
          "temporal resolution and potentially degrading model performance.")
    
    
# ================================================
# Exercise 3: exercise_convolution_freq_separation()
# ================================================
def generate_sine_superposition(len_signal=256, num_pairs=1000):
    """
    Generate signals by summing low- and high-frequency sinusoids with noise.

    Returns:
        t (ndarray): Time vector (len_signal,)
        X (ndarray): Input = low + high + noise, shape (num_pairs, len_signal)
        Y (ndarray): Target = high-frequency only, shape (num_pairs, len_signal)
    """
    t = np.linspace(0, 1, len_signal)
    X, Y = [], []
    for _ in range(num_pairs):
        f_low, p_low = np.random.uniform(2, 6), np.random.uniform(0.0, 1.0)
        f_high, p_high = np.random.uniform(12, 20), np.random.uniform(0.0, 1.0)
        noise = np.random.normal(0, 0.5, size=len_signal)
        x_low = np.sin(2 * np.pi * (f_low * t + p_low))
        y_high = 0.5 * np.sin(2 * np.pi * (f_high * t + p_high))
        X.append(x_low + y_high + noise)
        Y.append(y_high)

    return t, np.array(X), np.array(Y)

# Custom Dataset for (noisy_input, clean_target) signal pairs
class SineWaveDataset(Dataset):
    """Custom Dataset for paired (input, target) sine wave signals."""
    def __init__(self, X, Y):
        super().__init__()
        assert len(X) == len(Y), "Mismatched input and target lengths"
        self.X, self.Y = X, Y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

      
# Define a learnable high-pass filter using Conv1d
class LearnableHighpass(nn.Module):
    """Single-layer Conv1d model to learn a high-pass filter."""
    def __init__(self, kernel_size=21):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        return self.conv(x)

def exercise_convolution_freq_separation():
    """Exercise 3: Learning to Separate Low- and High-Frequency Components

    Notebook: PCPT_06_convolution.ipynb
    """
    # Generate dataset and visualize 24 input-output pairs
    t, X, Y = generate_sine_superposition()
    visualize_pairs_24(t, X, Y)

    # Convert to PyTorch tensors and add channel dimension: (batch, channels, length)
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (B, 1, T)
    Y_tensor = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)  # (B, 1, T)

    # Dataset and DataLoader
    dataset = SineWaveDataset(X_tensor, Y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model, loss, and optimizer
    torch.manual_seed(0)  # For reproducible results
    model = LearnableHighpass(kernel_size=21)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Training loop
    n_epochs = 50
    for epoch in range(n_epochs):
        total_loss = 0.0
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:>2}/50 | Avg. Loss: {total_loss / len(dataloader):.6f}")

    # Plot the learned filter kernel using a stem plot
    kernel_weights = model.conv.weight.data.numpy().flatten()
    plot_filter_kernel(kernel_weights)

    # Generate test signals
    t, X_test, Y_test = generate_sine_superposition(len_signal=256, num_pairs=7)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    Y_pred = model(X_test_tensor).detach().squeeze().numpy()

    # Visualize one example
    n = 6
    plot_signal_denoise(t, X_test[n], Y_test[n], Y_pred[n])

    # Visualize six additional examples
    visualize_further_examples_6(t, X_test, Y_test, Y_pred)
