"""
Module: libpcpt.unit04
Authors: Meinard Mueller, Johannes Zeitler, Sebastian Strahl, International Audio Laboratories Erlangen
License: The MIT license, https://opensource.org/licenses/MIT
This file is part of the PCPT Notebooks (https://www.audiolabs-erlangen.de/PCPT)
"""

import numpy as np
import torch
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

# Generate synthetic training data (input-output pairs)
def generate_training_pairs(num=100):
    torch.manual_seed(0)      # for reproducibility
    x = torch.rand(num) * 2   # random inputs in range [0, 2)   
    # Nonlinear target function with quadratic and oscillatory components + noise
    y = x + 2 * (x - 0.9)**2 + 0.3 * torch.cos(20 * x) + 0.1 * torch.randn(num)
    return x, y

# Function to visualize training data and model predictions
def plot_training_pairs(x_train, y_train, para_learn=None, para_reg=None):
    plt.figure(figsize=(4.5, 1.8))
    
    # Scatter plot of training data
    plt.scatter(x_train.numpy(), y_train.numpy(),
                color='black', marker='.', label='Training data')
    
    # Create evaluation grid
    x_val = torch.linspace(0, 2, 100)
    
    # Plot learned function (e.g., learned parameters from a model)
    if para_learn is not None:
        y_val = para_learn[0] + para_learn[1] * x_val + para_learn[2] * x_val ** 2
        plt.plot(x_val.numpy(), y_val.numpy(),
                 color='red', linewidth=2.0, linestyle='-', label='Learned function')
    
    # Plot reference polynomial regression (if provided)
    if para_reg is not None:
        y_val = para_reg[0] + para_reg[1] * x_val + para_reg[2] * x_val ** 2
        plt.plot(x_val.numpy(), y_val.numpy(),
                 color='blue', linewidth=2.0, linestyle='-', label='Polynomial regression')
    
    # Formatting and labels
    plt.xlabel("x", fontsize=8)
    plt.ylabel("y", fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.ylim(bottom=0)
    plt.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    

# ================================================
# Exercise 1: exercise_learning_rate()
# ================================================

# Regression using gradient descent
def regression_polynomial_second_order_via_gradient(x, y, learning_rate=1e-2, num_iterations=4000):
    """
    Fit a second-order polynomial y ≈ a + b*x + c*x^2 using gradient descent (PyTorch autograd).
    Args:
        x, y: torch.FloatTensor of shape [N]
        learning_rate: step size
        num_iterations: number of iterations
    Returns:
        a, b, c: learned parameters (floats)
        loss: final loss value (float)
    """
    torch.manual_seed(1)
    a = torch.randn((), requires_grad=True)
    b = torch.randn((), requires_grad=True)
    c = torch.randn((), requires_grad=True)

    for k in range(num_iterations):
        y_pred = a + b * x + c * x**2
        loss = (y_pred - y).pow(2).mean()
        loss.backward()

        with torch.no_grad():
            a -= learning_rate * a.grad
            b -= learning_rate * b.grad
            c -= learning_rate * c.grad
            a.grad = b.grad = c.grad = None

    return a.item(), b.item(), c.item(), loss.item()

# Closed-form regression solution
def regression_polynomial_second_order(x: torch.Tensor, y: torch.Tensor):
    """
    Fit a second-order polynomial to data (x, y) using closed-form linear regression.

    Parameters:
        x (torch.Tensor): 1D tensor of input values.
        y (torch.Tensor): 1D tensor of target values.

    Returns:
        a, b, c (float): Coefficients for the polynomial y ≈ a + b*x + c*x^2
        loss (float): Mean squared error (MSE) between predicted and target values.
    """
    # Construct design matrix: each row is [1, x_n, x_n^2]
    X_design = torch.vstack([torch.ones_like(x), x, x**2]).T

    # Solve normal equations: coef = (X^T X)^(-1) X^T y
    coef = torch.linalg.inv(X_design.T @ X_design) @ X_design.T @ y
    a, b, c = coef

    # Predict using fitted polynomial
    y_pred = a + b * x + c * x**2

    # Compute mean squared error
    loss = torch.mean((y_pred - y) ** 2)

    return a.item(), b.item(), c.item(), loss.item()

def exercise_learning_rate():
    """Exercise 1: Learning Rate

    Notebook: PCPT_04_grad.ipynb
    """
    # Run experiments
    # Prepare training data
    x, y = generate_training_pairs()

    # Vary learning rate
    print('\nChanging the learning rate (fixed num_iterations = 4000):')
    for rate in [1e-0, 1e-1, 1e-2, 1e-3]:
        a, b, c, loss = regression_polynomial_second_order_via_gradient(x, y, learning_rate=rate, num_iterations=4000)
        print(f'Learning rate {rate:>5.0e} | a: {a:7.4f}, b: {b:7.4f}, c: {c:7.4f}, loss: {loss:9.4f}')

    # Vary number of iterations
    print('\nChanging the number of iterations (fixed learning_rate = 1e-1):')
    for num in [100, 2000, 4000, 6000, 8000]:
        a, b, c, loss = regression_polynomial_second_order_via_gradient(x, y, learning_rate=1e-1, num_iterations=num)
        print(f'Iterations {num:4}         | a: {a:7.4f}, b: {b:7.4f}, c: {c:7.4f}, loss: {loss:9.4f}')

    # Vary number of iterations
    print('\nChanging the number of iterations (fixed learning_rate = 1e-2):')
    for num in [100, 2000, 4000, 6000, 8000]:
        a, b, c, loss = regression_polynomial_second_order_via_gradient(x, y, learning_rate=1e-2, num_iterations=num)
        print(f'Iterations {num:4}         | a: {a:7.4f}, b: {b:7.4f}, c: {c:7.4f}, loss: {loss:9.4f}')

    # Closed-form solution
    a_reg, b_reg, c_reg, loss_reg = regression_polynomial_second_order(x, y)
    print(f'\nClosed-form solution    | a: {a_reg:7.4f}, b: {b_reg:7.4f}, c: {c_reg:7.4f}, loss: {loss_reg:9.4f}')
    
    print('\nObservations / Conclusions:')
    print('* Very large learning rates (e.g., 1e+0) can cause divergence or instability.')
    print('* Moderate rates (e.g., 1e-1) may converge fast but risk overshooting or oscillating.')
    print('* Small rates (e.g., 1e-2) often ensure stable and effective convergence.')
    print('* Very small rates (e.g., 1e-3) are stable but slow, requiring many iterations.')
    print('* Choosing a suitable learning rate balances speed and stability.')
    print('* Gradient descent can closely match the closed-form solution if well-tuned.')
    print('* Note that what counts as a small or large learning rate depends on the model, data, and optimization')
    print('  landscape. For some problems, even 1e-3 may be too high, while for others it may be too low.')

# ================================================
# Exercise 2: exercise_gradient_accumulation()
# ================================================
def run_accumulation_demo(clear_grad=True, steps=10, learning_rate=0.1):
    x = torch.tensor(1.5)
    y = torch.tensor(2.5)
    a = torch.tensor(0.5, requires_grad=True)
    b = torch.tensor(0.5, requires_grad=True)

    a_vals = []
    b_vals = []
    loss_vals = []

    title = "With" if clear_grad else "Without"
    print(f"\n{title} gradient clearing:")
    print(f"Target function: loss = (a * x + b - y)**2 with x = {x.item()}, y = {y.item()}")    
    print(f"{'Step':>4s} | {'Loss':>10s} | {'a':>8s} | {'b':>8s} | {'grad_a':>8s} | {'grad_b':>8s}")
    print("-" * 60)

    for step in range(1, steps + 1):
        loss = (a * x + b - y) ** 2
        loss.backward()

        print(f"{step:4d} | {loss.item():10.6f} | {a.item():8.4f} | {b.item():8.4f} | {a.grad.item():8.4f} | {b.grad.item():8.4f}")

        a_vals.append(a.item())
        b_vals.append(b.item())
        loss_vals.append(loss.item())

        with torch.no_grad():
            a -= learning_rate * a.grad
            b -= learning_rate * b.grad

        if clear_grad:
            a.grad = None
            b.grad = None

    return a_vals, b_vals, loss_vals


def exercise_gradient_accumulation():
    """Exercise 2: Gradient Accumulation Pitfall

    Notebook: PCPT_04_grad.ipynb
    """
    # Run both cases
    a_accum, b_accum, _ = run_accumulation_demo(clear_grad=False)
    a_clear, b_clear, _ = run_accumulation_demo(clear_grad=True)

    print("\n") 

    # Plot parameter trajectories
    plt.figure(figsize=(7, 2.5))

    plt.subplot(1, 2, 1)
    plt.plot(a_accum, label='a (accumulated)', marker='o', markersize=4)
    plt.plot(b_accum, label='b (accumulated)', marker='x', markersize=4)
    plt.title("Without Gradient Clearing", fontsize=9)
    plt.xlabel("Iteration", fontsize=8)
    plt.ylabel("Parameter Value", fontsize=8)
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=8)

    plt.subplot(1, 2, 2)
    plt.plot(a_clear, label='a (cleared)', marker='o', markersize=4)
    plt.plot(b_clear, label='b (cleared)', marker='x', markersize=4)
    plt.title("With Gradient Clearing", fontsize=9)
    plt.xlabel("Iteration", fontsize=8)
    plt.ylabel("Parameter Value", fontsize=8)
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=8)

    plt.tight_layout()
    plt.show()

# ================================================
# Exercise 3: exercise_gradient_function2D()
# ================================================
def optimize_himmelblau(
    u: float = None,
    v: float = None,
    learning_rate: float = 1e-3,
    num_steps: int = 300,
    loss_tol: float = None,
    verbose: bool = True
):
    """
    Performs gradient descent on the Himmelblau function starting from (u, v).

    Parameters:
        u (float or torch.Tensor, optional): Initial value for u. If None, initialized randomly.
        v (float or torch.Tensor, optional): Initial value for v. If None, initialized randomly.
        learning_rate (float): Step size for gradient descent updates.
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

    for step in range(1, num_steps + 1):
        # Define the Himmelblau function
        loss = (u**2 + v - 11)**2 + (u + v**2 - 7)**2

        # Store loss and trajectory
        loss_value = loss.item()
        loss_history.append(loss_value)
        trajectory.append((u.item(), v.item()))

        # Verbose progress output
        if verbose and (step % 100 == 0 or step == num_steps):
            print(f"Step {step:4d} | u = {u.item():.4f}, v = {v.item():.4f}, loss = {loss_value:.6f}")

        # Backpropagation
        loss.backward()

        # Early stopping based on loss change
        if loss_tol is not None and prev_loss is not None:
            delta = abs(loss_value - prev_loss)
            if delta < loss_tol:
                if verbose:
                    print(f"Stopping early at step {step}: loss change = {delta:.6f} < loss_tol = {loss_tol}")
                break
        prev_loss = loss_value

        # Gradient descent update
        with torch.no_grad():
            u -= learning_rate * u.grad
            v -= learning_rate * v.grad
            u.grad = None
            v.grad = None

    return u.item(), v.item(), loss_history, trajectory

def plot_history(loss_history, title="Loss during Gradient Descent", color="blue"):
    """
    Plots the loss history over training iterations.

    Parameters:
        loss_history (list or array-like): List of loss values.
        title (str): Title for the plot.
        color (str): Color of the line plot.
    """
    if not loss_history:
        print("Warning: loss history is empty. Nothing to plot.")
        return

    plt.figure(figsize=(4, 1.5))
    plt.plot(loss_history, color=color)
    plt.xlabel("Iteration", fontsize=8)
    plt.ylabel("Loss", fontsize=8)
    plt.title(title, fontsize=9)
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=8)
    plt.tight_layout()
    plt.show()

def himmelblau(u, v):
    return (u**2 + v - 11)**2 + (u + v**2 - 7)**2

def plot_himmelblau_trajectory(trajectory, cmap='gray'):
    """
    Plots the Himmelblau function as a heatmap and overlays the optimization trajectory.

    Parameters:
        trajectory (list or np.ndarray): List of (u, v) tuples recorded during optimization.
        cmap (str): Matplotlib colormap to use for the heatmap.
    """
    # Create mesh grid
    u_vals = torch.linspace(-6, 6, 400)
    v_vals = torch.linspace(-6, 6, 400)
    U, V = torch.meshgrid(u_vals, v_vals, indexing="xy")
    Z = himmelblau(U, V)

    # Convert trajectory to NumPy array
    trajectory = np.array(trajectory)
    u_traj, v_traj = trajectory[:, 0], trajectory[:, 1]

    # Apply logarithmic normalization to color mapping
    norm = mcolors.LogNorm(vmin=Z.min() + 1e-2, vmax=Z.max())

    # Plot
    plt.figure(figsize=(5, 3))
    im = plt.imshow(Z, extent=[-6, 6, -6, 6], origin='lower', cmap=cmap, norm=norm, aspect='auto')
    im.set_clim(vmax=500)  # set upper color limit
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Function value (log scale)", fontsize=8)
    plt.plot(u_traj, v_traj, color='red', linewidth=2, label="Optimization path")
    plt.scatter(u_traj[0], v_traj[0], color='white', edgecolor='black', marker='o', label="Start")
    plt.scatter(u_traj[-1], v_traj[-1], color='red', marker='x', label="End")
    plt.xlabel("u", fontsize=8)
    plt.ylabel("v", fontsize=8)
    plt.title("Himmelblau Function and Optimization Path", fontsize=9)
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.legend(loc="upper right", fontsize=8)
    plt.grid(False)
    plt.tick_params(axis='both', labelsize=8)    
    plt.tight_layout()
    plt.show()
    
def exercise_gradient_function2D():
    """Exercise 3: Gradient Descent on a 2D Function

    Notebook: PCPT_04_grad.ipynb
    """
    # Example 1
    u0, v0 = 1.0, 1.0
    learning_rate = 0.0005
    print(f"Initialized at: u = {u0}, v = {v0}, learning rate = {learning_rate}")
    u, v, loss_history, trajectory = optimize_himmelblau(
        u=u0, v=v0, learning_rate=learning_rate, verbose=True
    )
    print(f"Final: u = {u:.4f}, v = {v:.4f}, final loss = {loss_history[-1]:.6f}")
    plot_history(loss_history)

    # Example 2
    u0, v0 = -1.0, -1.0
    learning_rate = 0.0005
    print(f"Initialized at: u = {u0}, v = {v0}, learning rate = {learning_rate}")
    u, v, loss_history, trajectory = optimize_himmelblau(
        u=u0, v=v0, learning_rate=learning_rate, verbose=True
    )
    print(f"Final: u = {u:.4f}, v = {v:.4f}, final loss = {loss_history[-1]:.6f}")
    plot_history(loss_history)

    # Example 3
    u0, v0 = -1.0, 1.0
    learning_rate = 0.0005
    loss_tol = 0.0001
    print(f"Initialized at: u = {u0}, v = {v0}, learning rate = {learning_rate}, loss_tol = {loss_tol}")
    u, v, loss_history, trajectory = optimize_himmelblau(
        u=u0, v=v0, learning_rate=learning_rate, verbose=True, loss_tol=loss_tol
    )
    print(f"Final: u = {u:.4f}, v = {v:.4f}, final loss = {loss_history[-1]:.6f}")
    plot_history(loss_history)
    
    plot_himmelblau_trajectory(trajectory)
