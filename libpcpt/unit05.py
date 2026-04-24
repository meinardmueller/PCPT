"""
Module: libpcpt.unit05
Authors: Meinard Mueller, Johannes Zeitler, Sebastian Strahl, International Audio Laboratories Erlangen
License: The MIT license, https://opensource.org/licenses/MIT
This file is part of the PCPT Notebooks (https://www.audiolabs-erlangen.de/PCPT)
"""

import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torchinfo import summary
from functools import partial

# ========= General Functions =========
def generate_training_pairs(num=100):
    torch.manual_seed(0)      # for reproducibility
    x = torch.rand(num) * 2   # random inputs in range [0, 2)   
    # Nonlinear target function with quadratic and oscillatory components + noise
    y = x + 2 * (x - 0.9)**2 + 0.3 * torch.cos(20 * x) + 0.1 * torch.randn(num)
    return x, y

# Function to visualize training data and model prediction
def plot_training_pairs_model(x_train, y_train, para=None, model=None):
    plt.figure(figsize=(4.5, 1.8))
    plt.scatter(x_train, y_train, color='black', marker='.', label='Training data')
    if para or model:
        x_val = torch.linspace(0, 2, 100)        
        if para:        
            y_val = para[0] + para[1] * x_val + para[2] * x_val ** 2
        if model:
            x_val_torch = x_val.unsqueeze(1)
            y_val = model(x_val_torch).detach().numpy()
        plt.plot(x_val, y_val, color='red', linewidth=2.0, linestyle='-', label='Learned function')
        
    # Formatting and labels
    plt.xlabel("x", fontsize=8)
    plt.ylabel("y", fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.ylim(bottom=0)
    plt.legend(loc='upper left', fontsize=8)
    plt.tight_layout()


# ================================================
# Exercise 1: exercise_activation_function()
#             exercise_activation_function_experiment()
# ================================================
def exercise_activation_function():
    """Exercise 1: Activation Functions

    Notebook: PCPT_05_nn.ipynb
    """
    # Define input range
    x = torch.linspace(-5, 5, 400)

    # Dictionary of activation functions with their formulas as titles
    activations = {
        "ReLU: max(0, x)": torch.relu(x),
        "Sigmoid: 1 / (1 + e^{-x})": torch.sigmoid(x),    
        "Tanh: (e^x - e^{-x}) / (e^x + e^{-x})": torch.tanh(x),
        "Leaky ReLU: x if x > 0 else 0.1 * x": F.leaky_relu(x, negative_slope=0.1)
    }

    # Create a figure with 2 rows and 2 columns
    fig, axes = plt.subplots(2, 2, figsize=(6.2, 3.5))
    axes = axes.flatten()  # flatten 2D array to 1D for easy iteration

    # Plot each activation function
    for ax, (title, y) in zip(axes, activations.items()):
        ax.plot(x.numpy(), y.numpy(), color='blue')
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Input", fontsize=8)
        ax.set_ylabel("Output", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True)

    # Adjust spacing
    plt.tight_layout()
    plt.show()


def training_loop(num_iterations, optimizer, model, loss_fn, x_train, y_train):
    for k in range(1, num_iterations + 1):        
        # Forward pass
        y_pred = model(x_train)
        loss = loss_fn(y_pred, y_train)            

        # Backward pass and parameter update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss every 1000 iterations
        if k % 1000 == 0:
            print(f"Iteration {k:4d} | Loss = {loss.item():.6f}", flush=True)

def exercise_activation_function_experiment():
    """Exercise 1: Activation Functions (Experiment)

    Notebook: PCPT_05_nn.ipynb
    """
    class NeuralNetTwoLayer(nn.Module):
        def __init__(self, act_fn=nn.ReLU(), dim_hidden=3):
            super().__init__()
            # First linear layer: input dimension 1 -> hidden dimension dim_hidden
            self.linear1 = nn.Linear(in_features=1, out_features=dim_hidden)
            # Activation function
            self.activation = act_fn
            # Second linear layer: hidden dimension dim_hidden -> output dimension 1
            self.linear2 = nn.Linear(in_features=dim_hidden, out_features=1)

        def forward(self, x):
            # Apply first linear transformation
            x = self.linear1(x)
            # Apply activation function
            x = self.activation(x)
            # Apply second linear transformation
            y = self.linear2(x)
            return y

    # Load training data and add dimension
    x, y = generate_training_pairs()
    x = x.unsqueeze(1)  # Shape: (*, 1)
    y = y.unsqueeze(1)  # Shape: (*, 1)

    # Try different activations with corresponding learning rates
    activations = {
        "ReLU": (nn.ReLU(), 1e-2),
        "Sigmoid": (nn.Sigmoid(), 1e-1),
        "Identity": (nn.Identity(), 1e-2),        
    #    "Tanh ": (nn.Tanh(), 1e-1),
    #    "LeakyReLU(0.1)": (nn.LeakyReLU(negative_slope=0.1), 1e-1)
    }

    for title, (act_fn, lr) in activations.items():
        torch.manual_seed(0)
        model = NeuralNetTwoLayer(act_fn=act_fn, dim_hidden=3)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        print(f"Activation function: {title}; Learning rate: {lr}")
        training_loop(4000, optimizer, model, nn.MSELoss(), x, y)
        plot_training_pairs_model(x.squeeze(), y.squeeze(), model=model)
        plt.show()

# ================================================
# Exercise 2: exercises_model_capacity()
# ================================================
def exercises_model_capacity():
    """Exercise 2: Exploring Model Capacity

    Notebook: PCPT_05_nn.ipynb
    """
# Define a four-layer feedforward neural network
    class NeuralNetFourLayer(nn.Module):
        def __init__(self, act_fn=nn.ReLU(), dim_hidden=10):
            """
            Four-layer fully connected neural network:
            Input (1D) -> Linear(1, H) -> act_fn -> Linear(H, H) -> act_fn
                      -> Linear(H, H) -> act_fn -> Linear(H, 1) -> Output (1D)

            Parameters:
            - act_fn: activation function (default: nn.ReLU())
            - dim_hidden: number of hidden units in each hidden layer
            """
            super().__init__()
            self.linear1 = nn.Linear(1, dim_hidden)
            self.activation1 = act_fn
            self.linear2 = nn.Linear(dim_hidden, dim_hidden)
            self.activation2 = act_fn
            self.linear3 = nn.Linear(dim_hidden, dim_hidden)
            self.activation3 = act_fn
            self.linear4 = nn.Linear(dim_hidden, 1)

        def forward(self, x):
            x = self.activation1(self.linear1(x))
            x = self.activation2(self.linear2(x))
            x = self.activation3(self.linear3(x))
            return self.linear4(x)

    # --- Generate synthetic training data ---
    torch.manual_seed(0)
    x = torch.linspace(0.0, 2.2, steps=16).unsqueeze(1)          # Shape: (16, 1)
    y = torch.cos(2 * torch.pi * x) + 0.1 * torch.randn_like(x)  # Add noise to cosine signal

    # --- Training setup ---
    lr = 0.05
    dim_hidden = 10
    act_fn = nn.ReLU()  # Try: nn.Sigmoid(), nn.Tanh(), nn.LeakyReLU()
    model = NeuralNetFourLayer(act_fn=act_fn, dim_hidden=dim_hidden)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # --- Model summary ---
    print("\nModel summary (using torchinfo):")
    print(summary(model, input_size=(1,), col_width=20))

    # --- Training ---
    title = "Four-Layer Network"
    print(f"\nTraining model: {title} | Activation: {act_fn.__class__.__name__} | Hidden dim: {dim_hidden} | Lr: {lr}")
    training_loop(num_iterations=8000, optimizer=optimizer, model=model, loss_fn=nn.MSELoss(), x_train=x, y_train=y)

    # --- Visualization ---
    x_val = torch.linspace(0.0, 2.3, steps=100).unsqueeze(1)
    y_val = model(x_val).detach().numpy()

    plt.figure(figsize=(4, 1.5))
    plt.scatter(x, y, c='black', s=10, label='Train')
    plt.plot(x_val, y_val, 'r-', lw=2, label='Learn')
    plt.grid(True)
    plt.legend(loc='lower right', fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)    
    plt.tight_layout()
    plt.show()

    # --- Conclusion ---
    print("\nConclusion: The model shows signs of overfitting: it fits the training data very \nwell but the learned function is not a clean sinusoid. With 4 layers and 10 hidden \nunits each, it has more capacity than necessary for this small dataset. As a result, \nit partially fits the noise in the data rather than capturing the underlying trend.")

# ================================================
# Exercise 3: exercise_optimization()
# ================================================
def himmelblau(u, v):
    return (u**2 + v - 11)**2 + (u + v**2 - 7)**2

def optimize_himmelblau(
    u: float = None,
    v: float = None,
    optimizer = None,
    num_steps: int = 300,
    loss_tol: float = None,
    verbose: bool = True
):
    """
    Performs gradient descent on the Himmelblau function starting from (u, v).

    Parameters:
        u (float or torch.Tensor, optional): Initial value for u. If None, initialized randomly.
        v (float or torch.Tensor, optional): Initial value for v. If None, initialized randomly.
        optimizer: The optimizer to be used. Should be a functoos.partial instance. 
        num_steps (int): Maximum number of iterations.
        loss_tol (float or None): Optional stopping threshold for change in loss.
                                  If the absolute change in loss is below this value,
                                  training stops early.
        verbose (bool): If True, print optimization progress every 100 steps and at the end.

    Returns:
        u_final (float): Final value of u.
        v_final (float): Final value of v.
        loss_history (list of float): Recorded loss values per iteration.
        trajectory (list of tuple): Sequence of (u, v) coordinates during optimization.
    """
    torch.manual_seed(0)

    # Initialize parameters
    if u is None:
        u = torch.randn((), requires_grad=True)
    else:
        u = torch.tensor(u, dtype=torch.float32, requires_grad=True)

    if v is None:
        v = torch.randn((), requires_grad=True)
    else:
        v = torch.tensor(v, dtype=torch.float32, requires_grad=True)

    loss_history = []
    trajectory = []
    prev_loss = None
    optimizer = optimizer([u, v])

    for step in range(1, num_steps + 1):
        # Evaluate Himmelblau function
        loss = himmelblau(u, v)

        # Store loss and trajectory
        loss_value = loss.item()
        loss_history.append(loss_value)
        trajectory.append((u.item(), v.item()))

        # Verbose progress output
        if verbose and (step % 100 == 0 or step == num_steps):
            print(f"Step {step:4d} | u = {u.item():.4f}, v = {v.item():.4f}, loss = {loss_value:.6f}")

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        # Early stopping based on loss change
        if loss_tol is not None and prev_loss is not None:
            delta = abs(loss_value - prev_loss)
            if delta < loss_tol:
                if verbose:
                    print(f"Stopping early at step {step}: loss change = {delta:.6f} < loss_tol = {loss_tol}")
                break
        prev_loss = loss_value

    return u.item(), v.item(), loss_history, trajectory

def plot_himmelblau_trajectories(trajectories, cmap='gray', xlim=(-6, 6), ylim=(-6, 6)):
    """
    Plots the Himmelblau function as a heatmap and overlays several optimization trajectories.

    Parameters:
        trajectories (dict): Dictionary of trajectories for different optimizers.
        cmap (str): Matplotlib colormap to use for the heatmap.
        xlim (tuple of float, optional): x-axis plot limits
        ylim (tuple of float, optional): y-axis plot limits
    """
    # Create mesh grid
    u_vals = torch.linspace(-6, 6, 400)
    v_vals = torch.linspace(-6, 6, 400)
    U, V = torch.meshgrid(u_vals, v_vals, indexing="xy")
    Z = himmelblau(U, V)

    # Apply logarithmic normalization to color mapping
    norm = mcolors.LogNorm(vmin=Z.min() + 1e-2, vmax=Z.max())

    # Plot
    plt.figure(figsize=(6, 3.6))
    im = plt.imshow(Z, extent=[-6, 6, -6, 6], origin='lower', cmap=cmap, norm=norm, aspect='auto')
    im.set_clim(vmax=500)  # set upper color limit
    cbar = plt.colorbar() 
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Function value (log scale)", fontsize=8)    

    for opt_key, trajectory, col in trajectories:
        trajectory = np.array(trajectory)
        u_traj, v_traj = trajectory[:, 0], trajectory[:, 1]
        plt.plot(u_traj, v_traj, color=col, linewidth=2, label=opt_key, marker='.')
    
    plt.scatter(u_traj[0], v_traj[0], color='white', edgecolor='black', marker='o', s=70, label="Start")
    plt.xlabel("u", fontsize=8)
    plt.ylabel("v", fontsize=8)
    #plt.title(f"Himmelblau Function and Different Optimizers (u0={u_traj[0]:.1f}, v0={v_traj[0]:.1f})",fontsize=9)
    plt.title(f"Himmelblau Function with Different Optimizers",fontsize=9)    
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend(loc="upper right", fontsize=8)
    plt.grid(False)
    plt.tick_params(axis='both', labelsize=8)      
    plt.tight_layout()
    plt.show()

def optimize_plot_himmelblau_trajectories_many(u, v, xlim, ylim, optimizer_list):
    """
    Runs optimization on the Himmelblau function for different optimizers and produces a plot
    showing the function and the optimization trajectories.

    Parameters:
        u (float or torch.Tensor): Initial value for u. If None, initialized randomly.
        v (float or torch.Tensor): Initial value for v. If None, initialized randomly.
        xlim (tuple of float): x-axis plot limits
        ylim (tuple of float): y-axis plot limits
        optimizer_list (list of tuples): list of tuples specifying different optimizers
    """
    trajectories = []

    for opt_key, opt, col in optimizer_list:
        _, _, _, trajectory = optimize_himmelblau(
            u=u, v=v, optimizer=opt, verbose=False, num_steps=100, loss_tol=1e-3,
        )
        trajectories.append((opt_key, trajectory, col))
    
    plot_himmelblau_trajectories(trajectories, xlim=xlim, ylim=ylim)

def exercise_optimization():
    """Exercise 3: Exploring Different Optimizers

    Notebook: PCPT_05_nn.ipynb
    """
    # Example 1
    u0, v0 = -1, 0.0
    xlim = (-4, -0.5)
    ylim = (-1, 5)

    optimizer_list = [
        ("SGD (momentum=0.8)", partial(torch.optim.SGD, lr=0.01, momentum=0.8), 'w'),    
        ("SGD (momentum=0.5)", partial(torch.optim.SGD, lr=0.01, momentum=0.5), 'r'),
        ("SGD (momentum=0.0)", partial(torch.optim.SGD, lr=0.01, momentum=0.0), 'k'),
    ]

    optimize_plot_himmelblau_trajectories_many(u0, v0, xlim, ylim, optimizer_list)

    # Example 2
    u0, v0 = 3.0, 0.0
    #xlim = (2, 5)
    xlim = (2.8, 3.8)
    ylim = (-3, 3)

    optimizer_list = [
        ("Adam", partial(torch.optim.Adam, lr=0.03), 'w'),    
        ("RMSprop", partial(torch.optim.RMSprop, lr=0.03), 'r'),        
        ("SGD", partial(torch.optim.SGD, lr=0.01, momentum=0.7), 'k'),
    ]

    optimize_plot_himmelblau_trajectories_many(u0, v0, xlim, ylim, optimizer_list)
