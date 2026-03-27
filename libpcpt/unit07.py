"""
Module: libpcpt.unit07
Authors: Meinard Mueller, Johannes Zeitler, Sebastian Strahl, International Audio Laboratories Erlangen
License: The MIT license, https://opensource.org/licenses/MIT
This file is part of the PCPT Notebooks (https://www.audiolabs-erlangen.de/PCPT)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchinfo import summary
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.datasets import make_blobs

# ========= Visualizations =========

def visualize_pairs_24(t, X, Y):
    """
    Visualize the first 24 noisy signals along with their class labels and frequencies.

    Args:
        t (np.ndarray): Time axis, shape (length_signal,)
        X (np.ndarray or Tensor): Noisy input signals, shape [N, length_signal] or [N, 1, length_signal]
        Y (np.ndarray or Tensor): Label information, e.g. [class_id, frequency], shape [N, 2]
    """
    n_rows, n_cols = 4, 6
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(7.5, 3.5), sharex=True, sharey=True)
    axs = axs.flatten()

    for n in range(n_rows * n_cols):
        x = X[n]
        y = Y[n]
        axs[n].plot(t, x, color='black', linewidth=0.5)
        axs[n].set_title(f"Class {int(y[0])} / {y[1]:.1f} Hz", fontsize=7, pad=0)
        axs[n].axis('off')

    plt.subplots_adjust(hspace=0.6, wspace=0.15)
    plt.suptitle('First 24 Training Examples', fontsize=9, y=1.02)
    plt.show()    

def compute_and_plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix", figsize=(4, 3)):
    """
    Compute and plot a confusion matrix with custom styling.

    Args:
        y_true (array-like): Ground truth class labels.
        y_pred (array-like): Predicted class labels.
        class_names (list of str): Names of each class.
        title (str): Plot title.
        figsize (tuple): Size of the matplotlib figure.
    
    Returns:
        cm (np.ndarray): The computed confusion matrix.
    """
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", ax=ax, colorbar=False)

    # Adjust annotation font sizes
    for text_row in disp.text_:
        for text in text_row:
            text.set_fontsize(8)

    # Style labels and layout
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Predicted label", fontsize=8)
    ax.set_ylabel("True label", fontsize=8)
    ax.tick_params(axis='x', labelrotation=45, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    plt.tight_layout()
    plt.show()

    return cm
        
def plot_signed_distance_vs_frequency(model, X_tensor, Y_tensor, Y_meta, bins, figsize=(6.2, 3)):
    """
    Evaluate model predictions and plot signed distance to class boundary vs. frequency.

    Args:
        model (torch.nn.Module): Trained model.
        X_tensor (torch.Tensor): Input tensor of shape [N, 1, T].
        Y_tensor (torch.Tensor): Ground truth class labels (LongTensor).
        Y_meta (np.ndarray): Metadata array of shape [N, D], column 1 must be frequency.
        bins (list or np.ndarray): Class frequency bin edges.
        figsize (tuple): Size of the plot.
    """
    # Step 1: Evaluate model
    model.eval()
    with torch.no_grad():
        logits = model(X_tensor)
        preds = torch.argmax(logits, dim=1)
        targets = Y_tensor

    # Step 2: Extract true frequencies
    frequencies = Y_meta[:, 1]  # Assuming 2nd column is frequency

    # Step 3: Compute signed distance to nearest boundary
    boundaries = np.array(bins[1:-1])  # drop 0 and inf
    def signed_distance(f, boundaries):
        diffs = f - boundaries
        nearest_idx = np.argmin(np.abs(diffs))
        return diffs[nearest_idx]
    
    signed_distances = np.array([signed_distance(f, boundaries) for f in frequencies])

    # Step 4: Identify misclassified samples
    wrong_mask = (preds != targets).numpy()

    # Step 5: Plot
    plt.figure(figsize=figsize)
    plt.scatter(frequencies, signed_distances, color='lightgray', edgecolor='none', alpha=0.7, label="Correct")
    plt.scatter(frequencies[wrong_mask], signed_distances[wrong_mask], color='crimson', edgecolor='black', label="Wrong", zorder=3)
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.6)

    plt.xlim(0, 13)
    plt.ylim(-1.2, 1.5)

    plt.xlabel("True Frequency (Hz)", fontsize=9)
    plt.ylabel("Signed Distance (Hz)", fontsize=9)
    plt.title("Signed Distance to Nearest Class Boundary vs. Frequency", fontsize=9)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.legend(fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def visualize_out_of_range_predictions(t, X, freqs, preds, probs=None, title="Out-of-Range Inputs"):
    """
    Visualize waveforms and predictions for up to 8 out-of-range frequency inputs in a 2x4 layout.
    """
    n_rows, n_cols = 2, 4
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(7.5, 3), sharex=True, sharey=True)
    axs = axs.flatten()

    for i in range(min(8, len(X))):
        axs[i].plot(t, X[i], color='black', linewidth=0.5)
        axs[i].set_ylim(-1.1, 2)
        axs[i].set_title(f"{freqs[i]:.1f} Hz → Class {preds[i]}", fontsize=7, pad=1.5)
        axs[i].axis('off')

        if probs is not None:
            prob_str = ", ".join(f"{p:.2f}" for p in probs[i])
            axs[i].text(0.01, 0.82, prob_str, fontsize=6, transform=axs[i].transAxes)

    for j in range(len(X), 8):
        axs[j].axis('off')

    plt.subplots_adjust(hspace=0.6, wspace=0.25)
    plt.suptitle(title, fontsize=9, y=1.02)
    plt.show()
 
# ================================================
# Exercise 1: exercise_binary_cross_entropy()
# ================================================
def exercise_binary_cross_entropy():
    """Exercise 1: Understanding Binary Cross-Entropy Loss

    Notebook: PCPT_07_classification.ipynb
    """
    # Define binary cross-entropy loss
    def binary_cross_entropy(p, y):
        eps = 1e-10  # Prevent log(0) by clipping p to a safe range defined by epsilon
        p = np.clip(p, eps, 1 - eps)
        return - (y * np.log(p) + (1 - y) * np.log(1 - p))

    # Define range of predicted probabilities (excluding 0 and 1)
    p_vals = np.linspace(0.001, 0.999, 400)

    # Compute BCE loss for y = 1 and y = 0
    loss_y1 = binary_cross_entropy(p_vals, y=1)
    loss_y0 = binary_cross_entropy(p_vals, y=0)

    # Plot the BCE loss
    plt.figure(figsize=(4.0, 2.5))
    plt.plot(p_vals, loss_y1, label='y = 1', color='red')
    plt.plot(p_vals, loss_y0, label='y = 0', color='blue')
    plt.xlabel("Predicted probability $p$", fontsize=8)
    plt.ylabel("Loss $\\mathcal{L}(p, y)$", fontsize=8)
    plt.title("Binary Cross-Entropy Loss", fontsize=9)
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.ylim(0, 7)  # Limit to emphasize steep divergence near 0 or 1
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.show()

# ================================================
# Exercise 2: exercise_softmax_multiclass()
# ================================================
def exercise_softmax_multiclass():
    """Exercise 2: Exploring Softmax and Cross-Entropy in Multiclass Classification

    Notebook: PCPT_07_classification.ipynb
    """
    # Define input examples with logits and corresponding targets
    examples = [
        {'logits': [0.0, 0.0, 0.0, 0.0], 'target': 0},    
        {'logits': [1.0, 0.0, 0.0, 0.0], 'target': 0},       
        {'logits': [1.0, 0.0, 0.0, 0.0], 'target': 1},       
        {'logits': [2.0, 0.0, 0.0, 0.0], 'target': 0},    
        {'logits': [2.0, 0.0, 0.0, 0.0], 'target': 1},     
        {'logits': [1.0, 2.0, 3.0, 4.0], 'target': 3},
        {'logits': [6.0, 1.0, 2.0, 1.0], 'target': 0},
        {'logits': [6.0, 1.0, 2.0, 1.0], 'target': 1},    
        {'logits': [6.0, 0.1, 2.0, 1.0], 'target': 1},     
        {'logits': [6.0, -1.1, 2.0, 1.0], 'target': 1}     
    ]

    # Header
    print(f"{'Example':<8} {'Logit vector z':<22} {'Probability vector p = softmax(z)':<35} {'Target':<6} {'Loss':<8}")
    print("-" * 81)

    losses = []

    # Loop over examples and compute softmax + cross-entropy manually
    for i, ex in enumerate(examples):
        logits = torch.tensor(ex['logits'])
        target = ex['target']

        # Compute softmax and log-softmax
        probs = F.softmax(logits, dim=0)
        log_probs = torch.log(probs)

        # Cross-entropy loss (negative log-probability of true class)
        loss = -log_probs[target]
        losses.append(loss.item())

        # Format and print
        logits_str = "[" + ", ".join(f"{z:.1f}" for z in logits) + "]"
        probs_str = "[" + ", ".join(f"{p:.4f}" for p in probs) + "]"
        print(f"{i+1:<8} {logits_str:<22} {probs_str:<35} {target:<6} {loss.item():<8.4f}")

    # Average cross-entropy loss
    mean_manual = sum(losses) / len(losses)
    print(f"\nTotal Cross-Entropy Loss (mean, manual): {round(mean_manual, 4)}")

    # Compare with PyTorch built-in cross-entropy
    logits_tensor = torch.tensor([ex['logits'] for ex in examples])
    targets_tensor = torch.tensor([ex['target'] for ex in examples])
    mean_builtin = F.cross_entropy(logits_tensor, targets_tensor).item()
    print(f"Total Cross-Entropy Loss (mean, F.cross_entropy): {round(mean_builtin, 4)}")
    
# ================================================
# Exercise 3: exercise_classification_FFT()
# ================================================
def generate_waveform(length_signal=256, num_samples=1000):
    """
    Generate a dataset of noisy sine or square waveforms with random frequency, phase, and noise.

    Each signal:
      - Prototype: sine (0) or square (1)
      - Frequency: sampled from [1, 13] Hz
      - Phase: sampled from [0, 1] (cycles)
      - Noise: Gaussian with std ∈ [0, 0.5]

    Returns:
        t (np.ndarray): Time axis, shape (length_signal,)
        X (np.ndarray): Noisy signals, shape (num_samples, length_signal)
        Y (np.ndarray): Metadata [prototype, frequency, phase, noise_std], shape (num_samples, 4)
    """
    t = np.linspace(0, 1, length_signal)
    X, Y = [], []

    for _ in range(num_samples):
        freq = np.random.uniform(1, 13)
        phase = np.random.uniform(0, 1)
        noise_std = np.random.uniform(0, 1)
        prototype = np.random.randint(0, 2)

        signal = np.sin(2 * np.pi * (freq * t - phase))
        if prototype == 1:
            signal = np.sign(signal)  # convert to square wave

        noisy = signal + np.random.normal(0.0, noise_std, size=length_signal)
        X.append(noisy)
        Y.append([prototype, freq, phase, noise_std])

    return t, np.array(X), np.array(Y)

def exercise_classification_FFT():
    """Exercise 3: Frequency Classification from FFT Features

    Notebook: PCPT_07_classification.ipynb
    """
    # ----------------------------
    # 1. Set seeds for reproducibility
    # ----------------------------
    np.random.seed(0)
    torch.manual_seed(0)

    # ----------------------------
    # 2. Generate Data and Compute FFT
    # ----------------------------
    # Generate waveforms with associated metadata
    t, X, Y = generate_waveform()

    # Compute magnitude of FFT and keep only the first 40 coefficients
    X_fft = np.abs(np.fft.rfft(X))[:, :40]  # shape: [N, 40]

    # Normalize each spectrum to have max = 1 (per sample)
    X_fft /= np.maximum(X_fft.max(axis=1, keepdims=True), 1e-8)

    # ----------------------------
    # 3. Visualize FFT Features
    # ----------------------------
    num_signals = 24
    num_cols = 6
    num_rows = num_signals // num_cols

    plt.figure(figsize=(7.5, 3.6))
    for i in range(num_signals):
        waveform_type = "sine" if Y[i, 0] == 0 else "square"
        freq = Y[i, 1]
        plt.subplot(num_rows, num_cols, i + 1)
        plt.plot(X_fft[i], color='black', linewidth=1)
        plt.title(f"{waveform_type} | {freq:.1f} Hz", fontsize=8)
        plt.axis('off')  # Hide axes for cleaner look
    plt.suptitle("First 24 Magnitude Normalized Spectra (40 bins)", fontsize=9)
    plt.subplots_adjust(top=0.86, hspace=0.4)
    plt.show()

    # ----------------------------
    # 4. Prepare Dataset and Labels
    # ----------------------------
    # Bin frequencies into 6 classes
    bins = [0, 2, 4, 6, 8, 10, np.inf]
    class_names = ["[0,2)", "[2,4)", "[4,6)", "[6,8)", "[8,10)", "[10,∞)"]
    Y_freq = Y[:, 1]
    Y_class = np.digitize(Y_freq, bins) - 1  # class labels in [0, 5]

    # Wrap in PyTorch tensors
    X_fft_tensor = torch.tensor(X_fft, dtype=torch.float32).unsqueeze(1)  # [N, 1, 40]
    Y_class_tensor = torch.tensor(Y_class, dtype=torch.long)              # [N]

    # PyTorch dataset
    class FFTDataset(Dataset):
        def __init__(self, X, Y):
            super().__init__()
            self.X = X
            self.Y = Y

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.Y[idx]

    dataset = FFTDataset(X_fft_tensor, Y_class_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # ----------------------------
    # 5. Define Simple Linear Classifier
    # ----------------------------
    class FFTClassifier(nn.Module):
        def __init__(self, input_size=40, num_classes=6):
            super().__init__()
            self.model = nn.Sequential(
                nn.Flatten(),                       # [B, 1, 40] → [B, 40]
                nn.Linear(input_size, num_classes)  # Linear classifier
            )

        def forward(self, x):
            return self.model(x)

    model = FFTClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.05)

    # Example input shape: batch size = 32, channels = 1, signal length = 40
    print("\nModel summary using torchinfo.summary:\n")
    print(summary(model, input_size=(32, 1, 40), col_width=20))

    # ----------------------------
    # 6. Training Loop
    # ----------------------------
    print("\nTraining model...\n")
    n_epochs = 10
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        correct = 0

        for x_batch, y_batch in dataloader:
            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y_batch).sum().item()

        acc = 100 * correct / len(dataloader.dataset)
        print(f"Epoch {epoch+1:2d}/{n_epochs} | Loss: {total_loss:7.4f} | Accuracy: {acc:.2f}%")

    # ----------------------------
    # 7. Evaluation on New Test Set
    # ----------------------------
    # Generate independent test set
    np.random.seed(10)
    t_test, X_test, Y_test = generate_waveform()

    # Preprocess test data
    X_test_fft = np.abs(np.fft.rfft(X_test))[:, :40]
    X_test_fft /= np.maximum(X_test_fft.max(axis=1, keepdims=True), 1e-8)
    Y_test_freq = Y_test[:, 1]
    Y_test_class = np.digitize(Y_test_freq, bins) - 1

    X_test_tensor = torch.tensor(X_test_fft, dtype=torch.float32).unsqueeze(1)
    Y_test_tensor = torch.tensor(Y_test_class, dtype=torch.long)

    # Predict and evaluate
    model.eval()
    with torch.no_grad():
        logits = model(X_test_tensor)
        preds = torch.argmax(logits, dim=1)
        test_accuracy = (preds == Y_test_tensor).float().mean().item()

    print(f"\nTest Accuracy: {test_accuracy:.2%}")

    # Plot confusion matrix
    compute_and_plot_confusion_matrix(
        Y_test_tensor.numpy(), preds.numpy(), class_names,
        title="Confusion Matrix"
    )

# ================================================
# Exercise 4: exercise_pointcloud_classification()
# ================================================
class PointDataset(Dataset):
    def __init__(self, X, Y):
        super().__init__()
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def generate_pointcloud(n_points_per_class=300, noise=0.15, seed=0):
    """
    Create a 3-class dataset:
    - Class 0 and 1: two intertwined spirals (more curves, tighter coils)
    - Class 2: a Gaussian blob near the center

    Returns:
        PointDataset
    """
    np.random.seed(seed)

    # --- Spiral A (class 0): More tightly wound spiral ---
    theta = np.linspace(0, 3.2 * np.pi, n_points_per_class)
    r = 0.4 * theta  # slightly scaled
    x0 = r * np.cos(theta)
    y0 = r * np.sin(theta)
    x0 += np.random.randn(n_points_per_class) * noise
    y0 += np.random.randn(n_points_per_class) * noise
    X0 = np.stack([x0, y0], axis=1)
    Y0 = np.zeros(n_points_per_class, dtype=int)

    # --- Spiral B (class 1): offset by π and with slight shift ---
    theta_b = theta + np.pi
    r_b = r
    x1 = r_b * np.cos(theta_b)
    y1 = r_b * np.sin(theta_b)
    x1 += np.random.randn(n_points_per_class) * noise - 0.5
    y1 += np.random.randn(n_points_per_class) * noise
    X1 = np.stack([x1, y1], axis=1)
    Y1 = np.ones(n_points_per_class, dtype=int)

    # --- Blob (class 2): near the origin ---
    X2, _ = make_blobs(n_samples=n_points_per_class,
                       centers=[[-3, -3]],
                       cluster_std=0.7)
    Y2 = np.full(n_points_per_class, 2, dtype=int)

    # Combine
    X_np = np.vstack([X0, X1, X2]).astype(np.float32)
    Y_np = np.concatenate([Y0, Y1, Y2])

    return PointDataset(X_np, Y_np)
    
def plot_pointcloud(dataset, title="Point Cloud"):
    """
    Plot a 2D point cloud from a PointDataset with exactly 3 classes.
    Uses colors from the 'coolwarm' colormap for visual consistency with decision regions.

    Args:
        dataset (PointDataset): Must have .X (N, 2) and .Y (N,) attributes.
        title (str): Title for the plot.
    """
    X = dataset.X.numpy()
    Y = dataset.Y.numpy()

    # Updated colormap access (no deprecation warning)
    cmap = matplotlib.colormaps['coolwarm']
    colors = [cmap(0.0), cmap(0.5), cmap(1.0)]  # Blue, grey, red
    labels = ['Class 1', 'Class 2', 'Class 3']

    plt.figure(figsize=(3.9, 3.2))
    for cls in range(3):
        mask = Y == cls
        plt.scatter(
            X[mask, 0], X[mask, 1],
            color=colors[cls],
            label=labels[cls],
            s=14, edgecolor='k', linewidth=0.3
        )
    
    plt.title(title, fontsize=9)
    #plt.xlabel("x", fontsize=8)
    #plt.ylabel("y", fontsize=8)
    plt.xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
    plt.ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)    
    plt.legend(loc='upper left', fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()    

def plot_decision_boundary(model, dataset, title="Model Decision Regions"):
    """
    Plot the decision regions of a trained model using the dataset's 2D inputs and class labels.
    Also computes classification accuracy and appends it to the plot title.

    Args:
        model: Trained PyTorch model returning logits.
        dataset: A PointDataset with .X (N, 2) and .Y (N,) tensors.
        title (str): Plot title prefix.
    """
    X = dataset.X.numpy()
    Y = dataset.Y.numpy()

    model.eval()
    with torch.no_grad():
        # Compute prediction accuracy
        X_tensor = torch.tensor(X, dtype=torch.float32)
        logits = model(X_tensor)
        preds = torch.argmax(logits, dim=1).numpy()
        accuracy = 100 * (preds == Y).sum() / len(Y)

        # Generate meshgrid for decision boundary
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                             np.linspace(y_min, y_max, 300))
        grid = np.c_[xx.ravel(), yy.ravel()]
        grid_tensor = torch.tensor(grid, dtype=torch.float32)
        logits_grid = model(grid_tensor)
        preds_grid = torch.argmax(logits_grid, dim=1).numpy()
        zz = preds_grid.reshape(xx.shape)

        # Plotting
        plt.figure(figsize=(3.9, 3.2))
        plt.contourf(xx, yy, zz, levels=3, alpha=0.3, cmap='coolwarm')
        plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='coolwarm', edgecolors='k', s=14)

        # Create manual legend
        legend_elements = [
            patches.Patch(facecolor=plt.cm.coolwarm(0.0), edgecolor='k', label='Class 1'),
            patches.Patch(facecolor=plt.cm.coolwarm(0.5), edgecolor='k', label='Class 2'),
            patches.Patch(facecolor=plt.cm.coolwarm(1.0), edgecolor='k', label='Class 3')
        ]
        plt.legend(handles=legend_elements, loc='upper left', fontsize=8)

        # Show accuracy in title
        plt.title(f"{title} (Accuracy: {accuracy:.2f}%)", fontsize=9)
        #plt.xlabel("x", fontsize=8)
        #plt.ylabel("y", fontsize=8)
        plt.xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
        plt.ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)        
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def exercise_pointcloud_classification():
    """Exercise 4: 3-Class Point Cloud Classification

    Notebook: PCPT_07_classification.ipynb

    Returns:
        model (nn.Module): Trained PyTorch model        
    """
    dataset = generate_pointcloud(n_points_per_class=300, noise=0.15, seed=0)
    batch_size = 64
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    plot_pointcloud(dataset)    
    class SimpleMLP(nn.Module):
        def __init__(self, input_dim=2, hidden_dim=32, num_classes=3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),

                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),

                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),

                nn.Linear(hidden_dim // 2, num_classes)
            )

        def forward(self, x):
            return self.net(x)
            
    torch.manual_seed(0)
    model = SimpleMLP(input_dim=2, hidden_dim=32, num_classes=3)
    print("\nModel summary using torchinfo.summary:\n")
    print(summary(model, input_size=(batch_size, 2), col_width=20))    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    n_epochs = 100
    for epoch in range(n_epochs):
        total_loss, correct, total = 0, 0, 0
        for x_batch, y_batch in dataloader:
            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

        if (epoch+1) % 10 == 0 or epoch == 0:
            acc = 100 * correct / total
            print(f"Epoch {epoch+1:3d}/{n_epochs} | Loss: {total_loss:7.4f} | Accuracy: {acc:.2f}%")

    # Compute prediction accuracy for entire training set 
    X = dataset.X.numpy()
    Y = dataset.Y.numpy()
    model.eval()
    with torch.no_grad():        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        logits = model(X_tensor)
        preds = torch.argmax(logits, dim=1).numpy()
        accuracy = 100 * (preds == Y).sum() / len(Y)
    print(f"\nFinal accuracy (entire training set; not averaged over batches): {accuracy:.2f}%")
        
    plot_decision_boundary(model, dataset)    
    
    return model 
