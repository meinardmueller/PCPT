"""
Module: libpcpt.unit08
Authors: Meinard Mueller, Johannes Zeitler, Sebastian Strahl, International Audio Laboratories Erlangen
License: The MIT license, https://opensource.org/licenses/MIT
This file is part of the PCPT Notebooks (https://www.audiolabs-erlangen.de/PCPT)
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import Dataset, DataLoader

# ========= General Functions =========

class PointDataset(Dataset):
    def __init__(self, X, Y):
        super().__init__()
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def generate_checkerboard_dataset(n_per_cell=100, noise=0.5, seed=0):
    """
    Generate a 2D checkerboard-style dataset with three alternating class labels,
    centered around the origin. Useful for studying overfitting and generalization.

    Args:
        n_per_cell (int): Number of points per grid cell.
        noise (float): Standard deviation of Gaussian noise added to points.
        seed (int): Random seed for reproducibility.

    Returns:
        PointDataset (torch.utils.data.Dataset): Dataset of (X, Y) pairs.
    """
    np.random.seed(seed)
    X_list, Y_list = [], []

    grid_size = 3         # 3x3 checkerboard
    spacing = 1.0         # Distance between centers

    # Offset to center grid at (0, 0)
    center_offset = (grid_size - 1) / 2.0

    for i in range(grid_size):
        for j in range(grid_size):
            label = (i + j) % 3
            x_center = (i - center_offset) * spacing
            y_center = (j - center_offset) * spacing

            x = x_center + noise * np.random.randn(n_per_cell)
            y = y_center + noise * np.random.randn(n_per_cell)

            X_cell = np.stack([x, y], axis=1)
            Y_cell = np.full(n_per_cell, label)

            X_list.append(X_cell)
            Y_list.append(Y_cell)

    X_all = np.vstack(X_list).astype(np.float32)
    Y_all = np.concatenate(Y_list)
    return PointDataset(X_all, Y_all)

def plot_decision_boundary(dataset, model=None, 
                           ax=None,
                           title=None,
                           figsize=(2.5, 2.5),
                           xlim_min=-2.5, xlim_max=2.5, 
                           ylim_min=-2.5, ylim_max=2.5):
    """
    Plot decision boundaries of a trained model on 2D input data, or just scatter plot of dataset if model is None.

    Args:
        dataset: A PointDataset with .X (N, 2) and .Y (N,) tensors.
        model (optional): Trained PyTorch model returning logits. If None, only data points are plotted.
        ax (matplotlib.axes.Axes or None): Optional axis for custom layout.
        title (str or None): Plot title.
        figsize (tuple): Size if no axis is passed.
        xlim_min, xlim_max, ylim_min, ylim_max (float): Axis limits.
    """
    X = dataset.X.numpy()
    Y = dataset.Y.numpy()

    # Use provided axis or create new one
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if model is not None:
        model.eval()
        with torch.no_grad():
            # Compute accuracy
            X_tensor = torch.tensor(X, dtype=torch.float32)
            logits = model(X_tensor)
            preds = torch.argmax(logits, dim=1).numpy()
            accuracy = 100 * (preds == Y).sum() / len(Y)

            # Generate meshgrid
            xx, yy = np.meshgrid(np.linspace(xlim_min, xlim_max, 300),
                                 np.linspace(ylim_min, ylim_max, 300))
            grid = np.c_[xx.ravel(), yy.ravel()]
            grid_tensor = torch.tensor(grid, dtype=torch.float32)
            logits_grid = model(grid_tensor)
            preds_grid = torch.argmax(logits_grid, dim=1).numpy()
            zz = preds_grid.reshape(xx.shape)

            ax.contourf(xx, yy, zz, levels=3, alpha=0.3, cmap='coolwarm')

            # Accuracy text in corner
            ax.text(xlim_max - 0.2, ylim_max - 0.2, f"Acc: {accuracy:.2f}%",
                    ha='right', va='top', fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.75, edgecolor='none'))

    # Always scatter data
    ax.scatter(X[:, 0], X[:, 1], c=Y, cmap='coolwarm', edgecolors='k', s=14)

    # Legend
    legend_elements = [
        patches.Patch(facecolor=plt.cm.coolwarm(0.0), edgecolor='k', label='Class 1'),
        patches.Patch(facecolor=plt.cm.coolwarm(0.5), edgecolor='k', label='Class 2'),
        patches.Patch(facecolor=plt.cm.coolwarm(1.0), edgecolor='k', label='Class 3')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8, frameon=True)

    if title:
        ax.set_title(title, fontsize=9)

    ax.set_xlim(xlim_min, xlim_max)
    ax.set_ylim(ylim_min, ylim_max)
    ax.set_xticks(np.arange(int(np.ceil(xlim_min)), int(np.floor(xlim_max)) + 1))
    ax.set_yticks(np.arange(int(np.ceil(ylim_min)), int(np.floor(ylim_max)) + 1))
    ax.tick_params(labelsize=8)
    ax.grid(True)
    ax.set_aspect("equal")

    if ax is None:
        plt.tight_layout()
        plt.show()
        
def plot_loss_and_accuracy(train_loss, val_loss, train_acc, val_acc, best_epoch, figsize=(6.2, 2.2)):
    """
    Plots training/validation loss and accuracy curves with best epoch marked.

    Args:
        train_loss (list): Training loss per epoch
        val_loss (list): Validation loss per epoch
        train_acc (list): Training accuracy per epoch (%)
        val_acc (list): Validation accuracy per epoch (%)
        best_epoch (int): Epoch index (1-based) of best validation performance
        figsize (tuple): Figure size
    """
    epochs = range(1, len(train_loss) + 1)
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    # --- Loss curve ---
    axs[0].plot(epochs, train_loss, label="Train Loss", color="tab:green", linewidth=1.5)
    axs[0].plot(epochs, val_loss, label="Val Loss", color="tab:red", linewidth=1.5)
    axs[0].axvline(best_epoch, color='gray', linestyle='--', linewidth=1, label=f"Best Epoch")
    axs[0].set_title("Loss", fontsize=9)
    axs[0].set_xlabel("Epoch", fontsize=8)
    axs[0].set_ylabel("Loss", fontsize=8)
    axs[0].tick_params(labelsize=8)
    axs[0].legend(fontsize=8, loc='upper right')
    axs[0].grid(True, linestyle=':', linewidth=0.5)

    # --- Accuracy curve ---
    axs[1].plot(epochs, train_acc, label="Train Acc", color="tab:green", linewidth=1.5)
    axs[1].plot(epochs, val_acc, label="Val Acc", color="tab:red", linewidth=1.5)
    axs[1].axvline(best_epoch, color='gray', linestyle='--', linewidth=1, label=f"Best Epoch")
    axs[1].set_title("Accuracy", fontsize=9)
    axs[1].set_xlabel("Epoch", fontsize=9)
    axs[1].set_ylabel("Accuracy (%)", fontsize=9)
    axs[1].tick_params(labelsize=8)
    axs[1].legend(fontsize=8, loc='lower right')
    axs[1].grid(True, linestyle=':', linewidth=0.5)

    plt.tight_layout()
    plt.show()

def plot_boundary_and_loss_lr(dataset, model, train_loss, lr_history, title="Training Set", figsize=(6.0, 2.5)):
    """Plot decision boundary (left) and training loss with learning rate (right)."""
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1, 2])

    # Left: decision boundary
    ax_left = fig.add_subplot(gs[0, 0])
    plot_decision_boundary(dataset, model=model, ax=ax_left, title=title)
    ax_left.tick_params(axis='both', labelsize=8)

    # Right: loss + LR
    ax_right = fig.add_subplot(gs[0, 1])
    epochs = range(1, len(train_loss) + 1)
    ax_right.plot(epochs, train_loss, label="Train Loss", color="tab:green", linewidth=1.5)
    ax_right.set_xlabel("Epoch", fontsize=8)
    ax_right.set_ylabel("Loss", fontsize=8, color="tab:green")
    ax_right.tick_params(axis='x', labelsize=8)  # x-axis font size
    ax_right.tick_params(axis='y', labelcolor="tab:green", labelsize=8)  # y-axis font size
    ax_right.set_title("Loss & Learning Rate", fontsize=9)
    ax_right.grid(True, linestyle=":", linewidth=0.5)

    # Learning rate (twin y-axis)
    ax_right_lr = ax_right.twinx()
    ax_right_lr.plot(epochs, lr_history, color="tab:blue", linestyle="--", linewidth=1, label="Learning Rate")
    ax_right_lr.set_ylabel("Learning Rate", fontsize=9, color="tab:blue")
    ax_right_lr.set_yscale("log")
    ax_right_lr.tick_params(axis='y', labelcolor="tab:blue", labelsize=8)

    # Merge legends
    lines1, labels1 = ax_right.get_legend_handles_labels()
    lines2, labels2 = ax_right_lr.get_legend_handles_labels()
    ax_right.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")

    plt.show()

# Extended training function with early stopping and optional scheduler
def train_model_extended(model, 
                dataloader, 
                val_dataloader=None, 
                criterion=None, 
                optimizer=None, 
                scheduler=None, 
                lr=0.01, 
                n_epochs=200, 
                logging_interval=50, 
                patience=None):
    """
    Trains a PyTorch model with optional validation monitoring, early stopping, and learning rate scheduling.

    Returns:
        model: Trained model (restored to best validation state if applicable)
        train_loss: List of training losses per epoch
        train_acc: List of training accuracies per epoch
        val_loss: List of validation losses per epoch (if val_dataloader is provided)
        val_acc: List of validation accuracies per epoch (if val_dataloader is provided)
        best_epoch: Epoch index (1-based) where validation loss was lowest
        lr_history: List of learning rates recorded at the end of each epoch
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    best_model_state = None
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0

    train_loss, train_acc = [], []
    val_loss, val_acc = [], []
    lr_history = []

    for epoch in range(n_epochs):
        # --- Training ---
        model.train()
        total_loss, correct, total = 0.0, 0, 0
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

        train_loss.append(total_loss)
        train_acc.append(100 * correct / total)

        # --- Validation ---
        if val_dataloader is not None:
            model.eval()
            v_loss, v_correct, v_total = 0.0, 0, 0
            with torch.no_grad():
                for x_val, y_val in val_dataloader:
                    logits = model(x_val)
                    loss = criterion(logits, y_val)
                    v_loss += loss.item()
                    preds = torch.argmax(logits, dim=1)
                    v_correct += (preds == y_val).sum().item()
                    v_total += y_val.size(0)
            val_loss.append(v_loss)
            val_acc.append(100 * v_correct / v_total)
        else:
            v_loss = total_loss  # fallback so ReduceLROnPlateau can still step
            val_loss.append(None)
            val_acc.append(None)

        # --- Scheduler step ---
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(v_loss)
            else:
                scheduler.step()

        # Record LR at end of epoch
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)

        # --- Logging ---
        if logging_interval is not None:
            if (epoch + 1) % logging_interval == 0 or epoch == 0:
                log = f"Epoch {epoch+1:3d}/{n_epochs} | Train Loss: {total_loss:7.4f} | Train Acc: {train_acc[-1]:6.2f}% | LR: {current_lr:.5g}"
                if val_dataloader:
                    log += f" | Val Loss: {v_loss:.4f} | Val Acc: {val_acc[-1]:6.2f}%"
                print(log)

        # --- Early stopping ---
        if val_dataloader and patience is not None:
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_model_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    if logging_interval is not None:
                        print(f"Early stopping triggered at epoch {epoch+1}.")
                    if best_model_state:
                        model.load_state_dict(best_model_state)
                    break

    return model, train_loss, train_acc, val_loss, val_acc, best_epoch, lr_history
        
# ================================================
# Exercise 1: exercise_tiny_model()
# ================================================
def exercise_tiny_model():
    """Exercise 1: 

    Notebook: PCPT_08_training.ipynb
    """
    dataset = generate_checkerboard_dataset(n_per_cell=20, noise=0.1, seed=0)

    # Define the MLP model
    class TinyMLP(nn.Module):
        def __init__(self, input_dim=2, hidden_dim=1, num_classes=3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes)
            )  
        def forward(self, x):
            return self.net(x)

    torch.manual_seed(0)
    batch_size = 16
    hidden_dim = 5
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TinyMLP(input_dim=2, hidden_dim=hidden_dim, num_classes=3)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Print a layer-wise model summary 
    # Note: uses random input and changes the global random state
    print(summary(model, input_size=(batch_size, 2), col_width=20))    
    print(f"\nTrainable parameters: {num_params}")
    print(f"Note: There are better solutions with fewer number of parameters!!!")    

    # Re-seed so training always starts from the same random state
    torch.manual_seed(0)
    lr_init = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)

    # Train
    model, train_loss, train_acc, val_loss, val_acc, best_epoch, lr_history = train_model_extended(
        model=model,
        dataloader=train_loader,
        val_dataloader=None,
        optimizer=optimizer,
        scheduler=None,
        n_epochs=50,
        logging_interval=10,
        patience=10
    )
    plot_decision_boundary(dataset, model=model, ax=None, title='', figsize=(2.5, 2.5))
 
# ================================================
# Exercise 2: exercise_random_seeds()
# ================================================
# Define the MLP model
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=16, num_classes=3):
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
        
def exercise_random_seeds():
    """Exercise 2: Exploring Softmax and Cross-Entropy in Multiclass Classification

    Notebook: PCPT_08_training.ipynb
    """ 
    # Generate three sets from same distribution but with different seeds
    dataset_train = generate_checkerboard_dataset(n_per_cell=20, noise=0.5, seed=0)
    dataset_val   = generate_checkerboard_dataset(n_per_cell=20, noise=0.5, seed=1)
    dataset_test  = generate_checkerboard_dataset(n_per_cell=20, noise=0.5, seed=2)

    # Dataloaders
    batch_size = 32
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(dataset_val,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(dataset_test,  batch_size=batch_size, shuffle=False)

    for seed in [0, 3, 18]:
        torch.manual_seed(seed)
        model = SimpleMLP(input_dim=2, hidden_dim=16, num_classes=3)

        # Train model
        model, train_loss, train_acc, val_loss, val_acc, best_epoch, lr_history = train_model_extended(
            model=model,
            dataloader=train_loader,
            val_dataloader=val_loader,
            lr=0.01,
            n_epochs=500,
            logging_interval=None,
            patience=20
        )

        # Evaluate on test set
        model.eval()
        test_loss, test_correct, test_total = 0.0, 0, 0
        with torch.no_grad():
            for x_test, y_test in test_loader:
                logits = model(x_test)
                loss = nn.CrossEntropyLoss()(logits, y_test)
                test_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                test_correct += (preds == y_test).sum().item()
                test_total += y_test.size(0)
        test_acc = 100 * test_correct / test_total

        # Print nicely formatted results
        print(f"Seed {seed}: "
              f"BestEp={best_epoch}, "
              f"TL={train_loss[best_epoch-1]:.3f}, TA={train_acc[best_epoch-1]:.1f}%, "
              f"VL={val_loss[best_epoch-1]:.3f}, VA={val_acc[best_epoch-1]:.1f}%, "
              f"TestL={test_loss:.3f}, TestA={test_acc:.1f}%")

        # Plot decision boundaries for this run
        fig, axs = plt.subplots(1, 3, figsize=(6.2, 2.5))
        plot_decision_boundary(dataset_train, model=model, ax=axs[0], title="Training Set")
        plot_decision_boundary(dataset_val,   model=model, ax=axs[1], title="Validation Set")
        plot_decision_boundary(dataset_test,  model=model, ax=axs[2], title="Test Set")
        plt.tight_layout()
        plt.show()
        
# ================================================
# Exercise 3: exercise_schedulers()
# ================================================
def exercise_schedulers():
    """Exercise 3: 

    Notebook: PCPT_08_training.ipynb
    """     
    # Exercise: Compare Multiple Configurations of Common LR Schedulers
    # Types: StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, OneCycleLR   
    # --- Settings ---
    base_lr = 0.1
    epochs = 30  # number of scheduler steps to visualize (per plot)

    # Dummy optimizer (no training needed)
    make_optimizer = lambda: optim.SGD(torch.nn.Linear(1, 1).parameters(), lr=base_lr)

    # Pretty params in titles
    fmt = lambda v: f"{v:.4g}" if isinstance(v, float) else str(v)

    # 3 configs per scheduler (kept meaningful and contrasting)
    scheduler_configs = {
        "StepLR": [
            {"step_size": 5,  "gamma": 0.5, "last_epoch": -1},
            {"step_size": 10, "gamma": 0.5, "last_epoch": -1},
            {"step_size": 10, "gamma": 0.2, "last_epoch": -1},
        ],
        "CosineAnnealingLR": [
            {"T_max": epochs, "eta_min": 0.0,           "last_epoch": -1},
            {"T_max": epochs, "eta_min": base_lr * 0.5, "last_epoch": -1},
            {"T_max": epochs, "eta_min": base_lr * 0.1, "last_epoch": -1},
        ],
        "CosineAnnealingWarmRestarts": [
            {"T_0": 10, "T_mult": 1, "eta_min": 0.0,               "last_epoch": -1},
            {"T_0": 5,  "T_mult": 2, "eta_min": base_lr * 0.1,     "last_epoch": -1},
            {"T_0": 7,  "T_mult": 3, "eta_min": base_lr * 0.5,     "last_epoch": -1},
        ],
        "OneCycleLR": [
            {"max_lr": base_lr, "total_steps": epochs, "pct_start": 0.3},
            {"max_lr": base_lr, "total_steps": epochs, "pct_start": 0.1},
            {"max_lr": base_lr, "total_steps": epochs, "pct_start": 0.5},
        ],
    }

    scheduler_classes = {
        "StepLR": optim.lr_scheduler.StepLR,
        "CosineAnnealingLR": optim.lr_scheduler.CosineAnnealingLR,
        "CosineAnnealingWarmRestarts": optim.lr_scheduler.CosineAnnealingWarmRestarts,
        "OneCycleLR": optim.lr_scheduler.OneCycleLR,
    }

    # --- Plot ---
    fig, axs = plt.subplots(len(scheduler_configs), 3, figsize=(6.2, 7), sharex=True, sharey=True)

    for row_idx, (sched_name, configs) in enumerate(scheduler_configs.items()):
        for col_idx, config in enumerate(configs):
            optimizer = make_optimizer()
            sched_class = scheduler_classes[sched_name]
            scheduler = sched_class(optimizer, **config)

            lrs = []
            for _ in range(epochs):
                optimizer.step()
                scheduler.step()
                lrs.append(optimizer.param_groups[0]['lr'])

            ax = axs[row_idx, col_idx]
            ax.plot(range(1, epochs + 1), lrs, marker='o', markersize=2.5)
            ax.set_ylim(bottom=0, top=base_lr+0.01)  # Keep y-limit consistent

            param_str = ", ".join(f"{k}={fmt(v)}" for k, v in config.items() if k != "last_epoch")
            ax.set_title(f"{sched_name}\n{param_str}", fontsize=7)
            ax.set_xlabel("Epoch", fontsize=7)
            ax.set_ylabel("LR", fontsize=7)
            ax.tick_params(labelsize=7)
            ax.grid(True, linestyle=':', linewidth=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    