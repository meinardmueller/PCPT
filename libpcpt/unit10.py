"""
Module: libpcpt.unit10
Authors: Meinard Mueller, Johannes Zeitler, Sebastian Strahl, International Audio Laboratories Erlangen
License: The MIT license, https://opensource.org/licenses/MIT
This file is part of the PCPT Notebooks (https://www.audiolabs-erlangen.de/PCPT)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ========= General Functions =========
def plot_data_boundary(ax, model, X, y, title="", add_legend=False, xlim=None, ylim=None):
    """
    Plot 2D points and the decision boundary of a binary classifier (logit output).
    Fixed colors, padding, and styles for consistent visuals.

    Args:
        ax: matplotlib axes
        model: torch.nn.Module producing a single logit per sample
        X: tensor (N,2)
        y: tensor (N,1) or (N,)
        title: str
        add_legend: bool
        xlim: tuple (xmin, xmax) or None
        ylim: tuple (ymin, ymax) or None
    """
    model.eval()

    # ---- Limits (auto if None; otherwise use provided) ----
    Xc = X.detach()
    yc = y.detach().view(-1)

    if xlim is None or ylim is None:
        x_min, x_max = Xc[:, 0].min().item(), Xc[:, 0].max().item()
        y_min, y_max = Xc[:, 1].min().item(), Xc[:, 1].max().item()
        x_pad = 0.08 * max(x_max - x_min, 1e-6)
        y_pad = 0.12 * max(y_max - y_min, 1e-6)
        x_lo, x_hi = x_min - x_pad, x_max + x_pad
        y_lo, y_hi = y_min - y_pad, y_max + y_pad
        if xlim is None: xlim = (x_lo, x_hi)
        if ylim is None: ylim = (y_lo, y_hi)

    # ---- Grid and model predictions (probabilities) ----
    gx, gy = np.meshgrid(
        np.linspace(xlim[0], xlim[1], 200),
        np.linspace(ylim[0], ylim[1], 200),
        indexing="xy"
    )
    grid = torch.tensor(
        np.c_[gx.ravel(), gy.ravel()],
        dtype=torch.float32,
        device=next(model.parameters()).device
    )
    with torch.no_grad():
        prob = torch.sigmoid(model(grid).view(-1)).detach().numpy().reshape(gx.shape)

    # ---- Background regions (fixed colors) ----
    ax.contourf(gx, gy, (prob >= 0.5).astype(float),
                levels=[-1, 0.5, 2], cmap="bwr", alpha=0.25, antialiased=True)

    # ---- Decision boundary at p=0.5 ----
    ax.contour(gx, gy, prob, levels=[0.5], colors="k", linewidths=1.25)

    # ---- Data points (fixed colors) ----
    sc = ax.scatter(Xc[:, 0], Xc[:, 1], c=yc.numpy(),
                    cmap="bwr", s=15, edgecolor="none")

    # ---- Aesthetics ----
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=8)

    if add_legend:
        handles, _ = sc.legend_elements()
        ax.legend(handles, ["Class 0", "Class 1"], loc="upper right", fontsize=8)
        
def plot_loss_curves(all_losses, epochs=None, labels=("0", "1"), figsize=(5, 2), ylim=None, ax=None):
    """
    Plot loss curves for training with/without regularization.
    all_losses: np.ndarray of shape (2, T)
        row 0 -> baseline (e.g., no BN / no dropout)
        row 1 -> regularized (e.g., with BN / with dropout)
    epochs: int or None
        if given, truncate curves to this many epochs
    labels: tuple of str
        names for the two curves
    ylim: tuple (ymin, ymax) or None
        if specified, sets y-axis limits
    ax: matplotlib Axes or None
        if provided, plot on this axes; otherwise create a new figure.
    """
    T = all_losses.shape[1]
    K = T if epochs is None else min(epochs, T)

    own_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        own_fig = True

    ax.plot(all_losses[0, :K], color="gray",  label=labels[0], linestyle="--", linewidth=1.5)
    ax.plot(all_losses[1, :K], color="black", label=labels[1], linestyle="-",  linewidth=1.2)

    ax.set_xlabel("Epoch", fontsize=8)
    ax.set_ylabel("Train loss", fontsize=8)
    ax.set_title(f"Loss", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.tick_params(labelsize=8)

    if ylim is not None:
        ax.set_ylim(*ylim)

    if own_fig:
        plt.tight_layout()
        plt.show()
    
def plot_regression_predictions(x, y, m_no, m_wd, D, xlim=(-1.1, 1.1), ylim=(-1.7, 1.7)):
    with torch.no_grad():
        xx = torch.linspace(xlim[0], xlim[1], 400).unsqueeze(1)
        #XX = poly_features(xx, D)
        XX = torch.cat([xx**k for k in range(D+1)], dim=1)
        pred_no = m_no(XX)
        pred_wd = m_wd(XX)
        # Example: true_curve = xx**2 - 0.5

    plt.figure(figsize=(5,2.2))
    plt.scatter(x.numpy(), y.numpy(), s=12, c='k', label='Data', alpha=0.9)
    plt.plot(xx.numpy(), pred_no.numpy(), lw=2.0, c='#0066aa',  alpha=0.6, label='no L2')
    plt.plot(xx.numpy(), pred_wd.numpy(), lw=2.0, c='#aa0000',  alpha=0.9, label='with L2')
    plt.legend(loc='lower right', fontsize=8)
    plt.xlabel('Inputs', fontsize=8)
    plt.ylabel('Targets', fontsize=8)
    plt.title("Polynomial Regression with and without L2 Regularization", fontsize=9)    
    plt.ylim(*ylim)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.show()
    
def plot_regression_coefficients(m_no, m_wd, title="Magnitude of Polynomial Coefficients"):
    with torch.no_grad():
        coefs_no = m_no.weight.squeeze().numpy()
        coefs_wd = m_wd.weight.squeeze().numpy()

    plt.figure(figsize=(5,1.8))
    plt.bar(range(len(coefs_no)), abs(coefs_no), color='#0066aa', alpha=0.6, label='no L2')
    plt.bar(range(len(coefs_wd)), abs(coefs_wd), color='#aa0000', alpha=0.9, label='with L2')
    plt.xlabel("Coefficient index", fontsize=8)
    plt.ylabel("|Coefficient|", fontsize=8)
    plt.title(title, fontsize=9)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()    
    
# --- Plot a loss matrix given (names, M) ---
def plot_loss_matrix(names, M, ax, title="", cmap="magma"):
    im = ax.imshow(M.detach().numpy(), cmap=cmap, aspect="auto", 
    interpolation="nearest", vmin=0, vmax=2)
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.grid(False)
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=8)

# --- Plot all signals (overlay) ---
def plot_six_signals(signals, L, figsize=(6.2, 2.5)):
    """Overlay plot of all signals with varied markers and line widths."""
    plt.figure(figsize=figsize)
    markers = ['o', '+', 'x', 's', '^', 'D']
    lws     = [2.0, 1.0, 1.2, 1.0, 1.2, 1.1]
    mss     = [5.0, 3.0, 5.0, 3.0, 4.0, 3.0]
    n = torch.arange(L)
    for i, (name, x) in enumerate(signals.items()):
        plt.plot(n, x.detach().numpy(),
                 marker=markers[i % len(markers)],
                 ms=mss[i % len(mss)],
                 lw=lws[i % len(lws)],
                 label=name)
    plt.title(f"Six Example Signals of Length L={L}", fontsize=9)
    plt.xlabel("Signal index", fontsize=8); plt.ylabel("Amplitude", fontsize=8)
    plt.tick_params(labelsize=8)
    plt.grid(True, alpha=0.3); plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.show()
    
# ================================================
# Exercise 1: exercise_bn_running_stats()
# ================================================
def exercise_bn_running_stats():
    """Exercise 1: 

    Notebook: PCPT_10_essential.ipynb
    """    
    # --- Data helpers ---
    def make_data(kind="Unimodal", N=400):
        if kind == "Unimodal":
            X = 1 + 1.5 * torch.randn(N, 1) 
        else:  # Bimodal
            X = torch.cat([0.5 * torch.randn(N//2,1) - 1.0,
                           1.2 * torch.randn(N//2,1) + 3.0], dim=0)
            X = X[torch.randperm(N)]
        return X

    # Run BN for given momentum/batch/epochs; return running mean/var curves (start at 0/1)
    def run_bn(X, momentum=0.1, bs=32, epochs=30):
        model = nn.BatchNorm1d(1, momentum=momentum, affine=False)
        loader = torch.utils.data.DataLoader(X, batch_size=bs, shuffle=True, drop_last=False)
        rm, rv = [model.running_mean.item()], [model.running_var.item()]
        for _ in range(epochs):
            model.train()
            for xb in loader:
                _ = model(xb)
            rm.append(model.running_mean.item()); rv.append(model.running_var.item())
        return rm, rv

    # --- Plot helpers ---
    def plot_momentum_sweep(X, kind, momenta=(0.01, 0.02, 0.1), bs=32, epochs=30):
        true_mean, true_var = X.mean().item(), X.var(unbiased=False).item()
        res = {m: run_bn(X, momentum=m, bs=bs, epochs=epochs) for m in momenta}

        fig, axs = plt.subplots(1, 3, figsize=(6.0, 1.6), constrained_layout=True)
        # left: histogram
        axs[0].hist(X.numpy().ravel(), bins=40, color="gray")
        axs[0].set_title(f"Dataset ({kind} Case)", fontsize=8)
        axs[0].set_xlabel("Sample value", fontsize=7); axs[0].set_ylabel("Count", fontsize=7)
        axs[0].tick_params(labelsize=6)

        # middle/right: running stats
        for m, (rm, rv) in res.items():
            axs[1].plot(rm, label=f"Momentum={m}")
            axs[2].plot(rv, label=f"Momentum={m}")
        axs[1].axhline(true_mean, color="black", ls="--", lw=1, label="True mean")
        axs[2].axhline(true_var,  color="black", ls="--", lw=1, label="True var")

        axs[1].set_title(f"Running Mean (B={bs})", fontsize=8)
        axs[2].set_title(f"Running Variance (B={bs})", fontsize=8)
        for ax in axs[1:]:
            ax.set_xlabel("Epoch", fontsize=7); ax.set_ylabel("Value", fontsize=7)
            ax.tick_params(labelsize=6); ax.legend(fontsize=6, loc="lower right")
        plt.show()

    def plot_batchsize_sweep(X, kind, momentum=0.02, bss=(4,16,64), epochs=30):
        true_mean, true_var = X.mean().item(), X.var(unbiased=False).item()
        res = {bs: run_bn(X, momentum=momentum, bs=bs, epochs=epochs) for bs in bss}

        fig, axs = plt.subplots(1, 3, figsize=(6.0, 1.6), constrained_layout=True)
        # left: histogram
        axs[0].hist(X.numpy().ravel(), bins=40, color="gray")
        axs[0].set_title(f"Dataset ({kind} Case)", fontsize=7)
        axs[0].set_xlabel("Sample value", fontsize=7); axs[0].set_ylabel("Count", fontsize=7)
        axs[0].tick_params(labelsize=6)

        # middle/right: running stats
        for bs, (rm, rv) in res.items():
            axs[1].plot(rm, label=f"B={bs}")
            axs[2].plot(rv, label=f"B={bs}")
        axs[1].axhline(true_mean, color="black", ls="--", lw=1, label="true mean")
        axs[2].axhline(true_var,  color="black", ls="--", lw=1, label="true var")

        axs[1].set_title(f"Running Mean (Momentum={momentum})", fontsize=7)
        axs[2].set_title(f"Running Variance (Momentum={momentum})", fontsize=7)
        for ax in axs[1:]:
            ax.set_xlabel("Epoch", fontsize=7); ax.set_ylabel("Value", fontsize=7)
            ax.tick_params(labelsize=6); ax.legend(fontsize=6, loc="lower right")
        plt.show()

    # --- Run both examples ---
    for kind in ["Unimodal", "Bimodal"]:
        X = make_data(kind)
        plot_momentum_sweep(X, kind, momenta=(0.01, 0.04, 0.1), bs=16, epochs=30)
        plot_batchsize_sweep(X, kind, momentum=0.04, bss=(4,16,64), epochs=30)

    # --- Conclusions ---
    print(
        "* With small batch sizes, the per-batch mean and variance fluctuate strongly,\n"
        "  so the running statistics look zigzaggy.\n"
        "* At the same time, more batches per epoch mean more updates, so the statistics\n"
        "  adapt faster even with the same momentum.\n"
        "* Large batches give smoother but fewer updates, making convergence steadier but\n"
        "  slower.\n"
        "* In practice, BatchNorm works best with reasonably large batches that balance\n"
        "  stability and adaptability."
    )

# ================================================
# Exercise 2: exercise_dropout_masks()
# ================================================
def exercise_dropout_masks():
    """Exercise 2: Visualizing Dropout Masks and Inverted Scaling

    Notebook: PCPT_10_essential.ipynb
    """ 
    torch.manual_seed(0)
    # --- Structured nput activations ---
    N, D = 8, 10
    X = torch.linspace(0.0, 1.0, steps=N*D).view(N, D)

    # --- Dropout probabilities ---
    ps = [0.25, 0.5, 0.75]
    cm = "inferno"

    def show(ax, data, title, cmap, clim):
        im = ax.imshow(
            data, cmap=cmap, aspect="auto", origin="upper",
            vmin=clim[0], vmax=clim[1]
        )
        ax.set_title(title, fontsize=8)
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=7)
        ax.set_xticks(range(data.shape[1]))
        ax.set_yticks(range(data.shape[0]))
        ax.tick_params(labelsize=6)
        ax.set_xlabel("Features", fontsize=7)
        ax.set_ylabel("Samples", fontsize=7)

    for p in ps:
        model = nn.Dropout(p=p)
        #print(f"Dropout layer used to create the mask: nn.Dropout(p={p})")

        # Build ONE mask first (independent of X):
        mask = model(torch.ones_like(X)) * (1 - p)

        # Use that exact mask to create the dropout output deterministically
        Y = X * mask / (1 - p)

        # Coherent color limits
        clim_in  = (0.0, 1.0)
        clim_out = (0.0, 1.0 / (1.0 - p))

        # --- Plot only this row (3 panels) ---
        fig, axs = plt.subplots(1, 3, figsize=(6, 1.6), constrained_layout=True)
        show(axs[0], X,    "Input X",               cm,      clim_in)
        show(axs[1], mask, f"Dropout Mask (p={p})", "gray_r", (0, 1))
        show(axs[2], Y,    "After Dropout",         cm,      clim_out)
        plt.show()    
        
# ================================================
# Exercise 3: exercise_custom_losses()
# ================================================
# --- Loss functions ---
def l1_loss(y_pred, y_true):
    return torch.mean(torch.abs(y_pred - y_true))

def corr_loss(y_pred, y_true, eps=1e-8):
    yp = y_pred - y_pred.mean()
    yt = y_true - y_true.mean()
    yp = yp / (yp.norm() + eps)
    yt = yt / (yt.norm() + eps)
    corr = torch.sum(yp * yt)   # cosine similarity on zero-mean, unit-norm signals
    return 1.0 - corr           # convert into loss, where lower numbers reflect higher correlation

def spectral_l2_fft(y_pred, y_target, n_fft=None):
    if n_fft is None: n_fft = y_pred.shape[-1]
    Mp = torch.fft.rfft(y_pred, n=n_fft, norm="ortho").abs()
    Mt = torch.fft.rfft(y_target, n=n_fft, norm="ortho").abs()
    return ((Mp - Mt)**2).mean()
    
def plot_conv_result(x, y_target, y_est, h_est, h_true=None,
                     loss_curve=None, loss_name="Loss",
                     figsize=(6, 1.6), show_legend=None):
    """
    Plot: [loss curve] | [x] | [kernels] | [y_est vs y_target]

    Args:
        x, y_target, y_est, h_est, h_true: tensors or numpy arrays (1D).
        h_true can be None (then only learned kernel is shown).
        loss_curve: list/1D array of scalars or None.
        loss_name: string for the loss title.
        figsize: figure size tuple.
        show_legend: bool or None. If True, draw legends (lower right). If None/False, skip legends.
    """
    # to numpy 1D
    to_np = lambda a: (a.detach().view(-1).numpy()
                       if hasattr(a, "detach") else np.asarray(a).reshape(-1))

    x_np     = to_np(x)
    y_t_np   = to_np(y_target)
    y_est_np = to_np(y_est)
    h_est_np = to_np(h_est)
    h_true_np = to_np(h_true) if h_true is not None else None

    fig, axes = plt.subplots(1, 4, figsize=figsize)
    lighter_red = (1.0, 0.6, 0.6)   # light red for learned/estimated curves

    # 1) Loss curve
    if loss_curve is None:
        axes[0].plot([], [])
    else:
        axes[0].plot(loss_curve, color="gray")
    axes[0].set_title(f"Training Loss ({loss_name})", fontsize=8)
    axes[0].set_xlabel("Epoch", fontsize=7)
    axes[0].set_ylabel("Loss", fontsize=7)

    # 2) Input x
    axes[1].plot(x_np, label="x (input)", color="black", linewidth=0.8)
    axes[1].set_title("Input Signal", fontsize=8)
    axes[1].set_xlabel("Signal index", fontsize=7)
    if show_legend:
        axes[1].legend(fontsize=6, loc="lower right")

    # 3) Kernel(s)
    axes[2].plot(h_est_np, label="Learned h", linestyle="-", color=lighter_red, linewidth=1.5)
    if h_true_np is not None:
        axes[2].plot(h_true_np, label="True h", color="black", linewidth=0.8)
    axes[2].set_title("Convolution Kernel", fontsize=8)
    axes[2].set_xlabel("Signal index", fontsize=7)
    if show_legend:
        axes[2].legend(fontsize=6, loc="lower right")

    # 4) Output vs target
    axes[3].plot(y_est_np, label="y_est", linestyle="-", color=lighter_red, linewidth=1.5)
    axes[3].plot(y_t_np, label="y_target", color="black", linewidth=0.8)
    axes[3].set_title("Output Comparison", fontsize=8)
    axes[3].set_xlabel("Signal index", fontsize=7)
    if show_legend:
        axes[3].legend(fontsize=6, loc="lower right")

    # ticks fontsize = 6
    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=6)

    plt.tight_layout()
    return fig, axes    

def exercise_custom_losses():
    """Exercise 3: Learning a Convolution Kernel with Different Loss Functions

    Notebook: PCPT_10_essential.ipynb
    """       
    torch.manual_seed(0)
    # ----- Model -----
    class KernelLearner(nn.Module):
        """Learns a 1D convolution kernel of size K."""
        def __init__(self, K):
            super().__init__()
            self.conv = nn.Conv1d(
                in_channels=1, out_channels=1,
                kernel_size=K, bias=False, padding="same"
            )
        def forward(self, x):
            return self.conv(x)

    # -------- Examples --------
    # Example 1: random input, sinusoidal output, no true kernel
    L1, K1 = 121, 121
    x1 = torch.randn(L1).view(1, 1, -1)
    t1 = torch.linspace(0, 1, L1)
    y1 = torch.sin(2 * torch.pi * 3 * t1).view(1, 1, -1)
    h1 = None

    # Example 2: random input, random kernel, output via convolution plus noise
    L2, K2 = 16, 7
    x2 = torch.randn(L2).view(1, 1, -1)
    h2 = torch.randn(K2) * 0.2
    y2 = F.conv1d(x2, h2.view(1, 1, -1), padding="same")
    y2 = y2 + 0.1 * torch.randn_like(y2)

    examples = [
        ("Example 1: random input, sinusoidal output, no true kernel", x1, y1, h1, K1),
        ("Example 2: random input, random kernel, output via convolution plus noise", x2, y2, h2, K2),
    ]

    # --- Losses to compare (reuse your already-defined funcs) ---
    losses = [
        ("MSE Loss", F.mse_loss),
        ("L1 Loss",  l1_loss),
        ("Correlation Loss", corr_loss),
        ("Spectral Loss", spectral_l2_fft),
    ]

    # --- Train & plot per example and loss ---
    epochs = 100
    for ex_name, x, y_target, h_true, K in examples:
        print("\n" + "="*73)
        print(ex_name)
        print("="*73)
        
        for loss_name, loss_fn in losses:
            torch.manual_seed(0)
            model = KernelLearner(K)
            opt = torch.optim.Adam(model.parameters(), lr=0.05)
            loss_curve = []
            for _ in range(epochs):
                y_hat = model(x)
                loss = loss_fn(y_hat, y_target)
                opt.zero_grad(); loss.backward(); opt.step()
                loss_curve.append(loss.item())

            with torch.no_grad():
                y_est = model(x)
                h_est = model.conv.weight.view(-1)

            fig, axes = plot_conv_result(
                x, y_target, y_est, h_est,
                h_true=h_true,
                loss_curve=loss_curve,
                loss_name=f"{loss_name}",
                show_legend=True
            )
            plt.show()
