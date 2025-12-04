"""
Module: libpcpt.unit03
Authors: Meinard Mueller, Johannes Zeitler, Sebastian Strahl, International Audio Laboratories Erlangen
License: The MIT license, https://opensource.org/licenses/MIT
This file is part of the PCPT Notebooks (https://www.audiolabs-erlangen.de/PCPT)
"""

import torch
import matplotlib.pyplot as plt

# ================================================
# Exercise 1: exercise_image_tensor()
# ================================================
def exercise_image_tensor():
    """Exercise 1: Preprocessing Grayscale Image Tensors

    Notebook: PCPT_03_tensor.ipynb
    """
    # Create a random (8, 8) grayscale image tensor with pixel values in [0, 255]
    torch.manual_seed(42)  # For reproducibility
    img_int = torch.randint(low=0, high=256, size=(8, 8), dtype=torch.int32)

    print("=== Original Integer Tensor ===")
    print(img_int)
    print("\n--- Tensor Attributes ---")
    print(f"Shape: {img_int.shape}")
    print(f"Dtype: {img_int.dtype}")
    print(f"Device: {img_int.device}")
    print(f"Total number of elements: {img_int.numel()}")

    # Convert to float32 and normalize to [0.0, 1.0]
    img_float = img_int.to(dtype=torch.float32) / 255.0
    print("\n=== Normalized Float Tensor ===")
    print(img_float)
    print("\n--- Normalized Tensor Statistics ---")
    print(f"Dtype: {img_float.dtype}")
    print(f"Mean pixel value: {img_float.mean():.4f}")
    print(f"Standard deviation: {img_float.std():.4f}")

# ================================================
# Exercise 2: exercise_tensor_properties()
# ================================================    
def exercise_tensor_properties():
    """Exercise 2: Exploring Tensor Shapes and Views

    Notebook: PCPT_03_tensor.ipynb
    """
    # Create tensor with shape (2, 1, 1, 3)
    x = torch.zeros((2, 1, 1, 3))
    print(f"Original tensor x:\n{x}")
    print(f"Shape: {x.shape}, Dimensions: {x.dim()}, Total elements: {x.numel()}")

    # Remove singleton dimensions
    y = x.squeeze()
    print(f"After squeeze(): shape = {y.shape}")

    # Add a new dimension in the middle
    z1 = y.unsqueeze(1)
    print(f"After unsqueeze(1): shape = {z1.shape}")

    # Add dimensions using None (equivalent to unsqueeze)
    z2 = y[:, None, None, :]
    print(f"Using indexing with None: shape = {z2.shape}")

    # Flatten and reshape
    x_flat = x.view(-1)
    print(f"Flattened tensor shape: {x_flat.shape}")
    z = x_flat.view(3, 2)
    print(f"Reshaped x to (3, 2) to obtain z: z.shape = {z.shape}")

    # Modify original tensor and see the effect on views
    x[0, 0, 0, 0] = 1
    print(f"Tensor z after modifying x[0,0,0,0] = 1:\n{z}")
  
# ================================================
# Exercise 3: exercise_eigen_pca()
# ================================================   
def exercise_eigen_pca():
    """Exercise 3: Principal Directions via Eigen Decomposition

    Notebook: PCPT_03_tensor.ipynb
    """
    # Set parameters and random seed
    N = 200
    torch.manual_seed(42)

    # Transformation matrix for anisotropic scaling and shearing
    M = torch.tensor([[2.0, -1.0],
                      [1.0,  0.5]])

    # Generate and transform Gaussian data
    x_raw = torch.randn(N, 2)
    x = x_raw @ M.T  # Right-multiply to apply transformation

    # Center the data
    x_mean = x.mean(dim=0, keepdim=True)
    x_centered = x - x_mean

    # Verify that mean is approximately zero
    print(f"Mean after centering (should be close to 0):\n{x_centered.mean(dim=0)}\n")

    # Compute empirical covariance matrix
    C = x_centered.T @ x_centered / (N - 1)

    # Compute eigenvalues and eigenvectors
    eigvals, eigvecs = torch.linalg.eigh(C)
    eigvals = eigvals.real
    eigvecs = eigvecs.real

    #  Print results
    print("Covariance matrix C:")
    print(C.round(decimals=4), "\n")

    print("Eigenvalues (variances along principal directions):")
    print(eigvals.round(decimals=4), "\n")

    print("Eigenvectors (columns = principal directions):")
    print(eigvecs.round(decimals=4), "\n")
    
    # Simple visualization of centered data and principal directions
    plt.figure(figsize=(3, 3))
    plt.scatter(x_centered[:, 0], x_centered[:, 1], color='black', s=5)
    colors = ['red', 'cyan']
    for i in range(2):
        v = eigvecs[:, i] * eigvals[i].sqrt()
        plt.plot([0, v[0]], [0, v[1]], color=colors[i], linewidth=2)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)        
    plt.axis("equal")
    plt.show()    
