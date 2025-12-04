"""
Module: libpcpt.unit09
Authors: Meinard Mueller, Johannes Zeitler, Sebastian Strahl, International Audio Laboratories Erlangen
License: The MIT license, https://opensource.org/licenses/MIT
This file is part of the PCPT Notebooks (https://www.audiolabs-erlangen.de/PCPT)
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from scipy.signal import find_peaks

# ----------------------- Visualization -----------------------
def plot_io(x, y, title="", ax=None, figsize=(3.3, 1.8), xlim=None, offset=0.1):
    """Stem plot with input/output slightly shifted to avoid overlap."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    n_x = np.arange(len(x))
    n_y = np.arange(len(y)) + offset  # shift output positions

    # Input: black
    in_stem = ax.stem(n_x, x, linefmt='black', markerfmt='o', basefmt=' ', label='x[n]')
    in_stem.markerline.set_markersize(4)
    in_stem.stemlines.set_linewidth(1)

    # Output: red
    out_stem = ax.stem(n_y, y, linefmt='red', markerfmt='o', basefmt=' ', label='y[n]')
    out_stem.markerline.set_markersize(4)
    out_stem.stemlines.set_linewidth(1)

    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Index n", fontsize=8)
    ax.tick_params(axis='both', labelsize=8)
    ax.grid(True, linewidth=0.5)
    if xlim:
        ax.set_xlim(xlim)
    ax.legend(fontsize=8)
    return ax

def visualize_training_pairs(t, X, Y, n_rows=4, n_cols=6, title="First 24 Training Pairs"):
    """
    Visualize the first N = n_rows * n_cols target–input pairs as small multiples.
    Plot order: INPUT first (gray, thicker), then TARGET (black, thinner).
    Accepts numpy arrays or torch tensors with shapes:
      t:    [seq] (or None → uses range(seq))
      X, Y: [N, seq] or [N, seq, 1] or [N, 1, seq]
    """
    # to numpy (supports torch tensors)
    def _to_numpy(a):
        if isinstance(a, torch.Tensor):
            return a.detach().numpy()
        return np.asarray(a)

    # squeeze to [N, seq]
    def _squeeze_to_2d(Z):
        if Z.ndim == 3:
            if Z.shape[-1] == 1:  # [N, seq, 1]
                Z = Z[..., 0]
            elif Z.shape[1] == 1:  # [N, 1, seq]
                Z = Z[:, 0, :]
        return Z

    X = _to_numpy(X); Y = _to_numpy(Y)
    t = _to_numpy(t) if t is not None else None
    X = _squeeze_to_2d(X); Y = _squeeze_to_2d(Y)

    assert X.ndim == 2 and Y.ndim == 2 and X.shape == Y.shape, "X and Y must be [N, seq_len] with same shape."
    N, L = X.shape
    if t is None:
        t = np.arange(L, dtype=float)
    else:
        t = t.reshape(-1); assert t.shape[0] == L, "t must match sequence length."

    N_show = min(n_rows * n_cols, N)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6, 2.8), sharex=True, sharey=True)
    axs = axs.flatten()

    for n in range(N_show):
        ax = axs[n]
        # INPUT first (gray, thicker), then TARGET (black, thinner)
        ax.plot(t, X[n], linewidth=1.2, color='gray')   # input
        ax.plot(t, Y[n], linewidth=0.8, color='black')  # target
        ax.axis('off')

    for n in range(N_show, len(axs)):
        axs[n].axis('off')

    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.suptitle(title, fontsize=9)
    plt.show()

# ----------------------- Helpers -----------------------
def _gauss(L, center, sigma):
    """Unit-peak Gaussian (length L), centered at 'center' (float), std=sigma (samples)."""
    n = np.arange(L, dtype=float)
    return np.exp(-0.5 * ((n - center) / float(sigma))**2)

def _sample_peaks(seq_len, k, min_gap, rng):
    """Sample k peak positions with >= min_gap spacing (grid-based sampler)."""
    if min_gap < 1:
        min_gap = 1
    offset = int(rng.integers(0, min_gap))  # random phase so peaks don't always align
    candidates = np.arange(offset, seq_len, min_gap)
    k = min(k, len(candidates))
    return np.sort(rng.choice(candidates, size=k, replace=False)).tolist()

def _gauss_kernel1d(sigma, radius):
    """1D Gaussian kernel normalized to sum=1; radius in samples (kernel size = 2*radius+1)."""
    if sigma is None or sigma <= 0 or radius < 1:
        return np.array([1.0], dtype=float)
    n = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-0.5 * (n / float(sigma))**2)
    k /= np.sum(k)
    return k

# ----------------------- Dataset (fast decays, softer ascents; targets binary by default) -----------------------
class NoveltyCurveDataset(Dataset):
    """
    (X, Y) for spike detection with **fast, variable exponential decays** in X,
    and *less sudden* ascents (small Gaussian smoothing applied to inputs before noise).

    Y (targets): by default **binary** spikes at change locations (values in {0,1}).
                 If `sigma_target` is set to a positive value, each spike is replaced by
                 a unit-peak Gaussian of width `sigma_target`. (When `sigma_target=None`,
                 targets are binary regardless of `combine`.)

    X (inputs) : for each spike at position c:
        - Draw amplitude A ~ Uniform(amp_range)
        - Draw time constant tau ~ LogUniform(tau_range), r = exp(-1/tau)
        - Tail: A * r**(n - c) for n >= c
        - Optional onset softening over a short window (ramp + jitter)
        - Convolve full sequence with a small Gaussian kernel:
            sigma_smooth ~ Uniform(smooth_sigma_range), radius = smooth_radius
          (softens ascents without overly blurring decays)
        - Add Gaussian noise and scale per sequence so max(X) = x_max.

    Most important knobs:
      - x_max           : global max of each input sequence after scaling (e.g., 2.0)
      - min_gap         : minimum distance between spike positions (samples)
      - amp_range       : per-spike amplitude range for inputs (A)
      - tau_range       : (tau_min, tau_max) in samples — small → fast decay; sampled log-uniformly
      - sigma_target    : target Gaussian width (samples). Use None → binary targets.
      - noise_std       : Gaussian noise std added to inputs
      - smooth_sigma_range / smooth_radius : small Gaussian blur for inputs (soften ascents)

    Shapes:
      X: (N, T, 1), Y: (N, T, 1)
    """
    def __init__(self,
                 n_samples=800,
                 seq_len=128,
                 x_max=2.0,
                 min_gap=10,
                 amp_range=(0.6, 1.6),
                 tau_range=(2.0, 5.0),         # fast, variable decays
                 sigma_target=None,            # targets binary by default
                 noise_std=0.02,
                 n_peaks_range=(2, 6),
                 combine="max",
                 # onset softening (kept small; disable via rise_len_rng=(1,1))
                 rise_len_rng=(2, 6),
                 rise_jitter_std=0.05,
                 # small Gaussian smoothing to soften ascents in inputs
                 smooth_sigma_range=(0.5, 2.0),
                 smooth_radius=4,
                 seed=0):
        super().__init__()
        assert combine in ("max", "sum")
        self.seq_len = int(seq_len)
        self.x_max = float(x_max)
        self.min_gap = int(min_gap)
        self.amp_range = tuple(map(float, amp_range))
        self.tau_range = tuple(map(float, tau_range))
        self.sigma_target = None if sigma_target is None else float(sigma_target)
        self.noise_std = float(noise_std)
        self.n_peaks_range = tuple(map(int, n_peaks_range))
        self.combine = combine
        self.rise_len_rng = tuple(map(int, rise_len_rng))
        self.rise_jitter_std = float(rise_jitter_std)
        self.smooth_sigma_range = None if smooth_sigma_range is None else tuple(map(float, smooth_sigma_range))
        self.smooth_radius = int(smooth_radius)

        rng = np.random.default_rng(seed)
        X_list, Y_list = [], []
        log_tmin, log_tmax = np.log(self.tau_range[0]), np.log(self.tau_range[1])

        for _ in range(n_samples):
            # 1) spike positions with spacing
            K = int(rng.integers(self.n_peaks_range[0], self.n_peaks_range[1] + 1))
            centers = _sample_peaks(self.seq_len, K, self.min_gap, rng)

            # 2) targets Y
            y = np.zeros(self.seq_len, dtype=float)
            if self.sigma_target is None:
                # Binary targets (OR of spikes); ignore 'combine' to keep {0,1}
                for c in centers:
                    y[c] = 1.0
            else:
                for c in centers:
                    bump = _gauss(self.seq_len, c, self.sigma_target)
                    y = np.maximum(y, bump) if self.combine == "max" else (y + bump)

            # 3) inputs X: exponential decays + optional softened onset
            x = np.zeros(self.seq_len, dtype=float)
            for c in centers:
                A   = rng.uniform(self.amp_range[0], self.amp_range[1])  # amplitude
                tau = np.exp(rng.uniform(log_tmin, log_tmax))            # tau ~ log-uniform
                r   = np.exp(-1.0 / tau)                                 # 0<r<1
                n = np.arange(c, self.seq_len, dtype=float)
                tail = A * (r ** (n - c))

                # small onset softening (ramp + jitter)
                L_rise = int(rng.integers(self.rise_len_rng[0], self.rise_len_rng[1] + 1))
                L_rise = max(1, min(L_rise, self.seq_len - c))
                if L_rise > 0:
                    k = np.arange(L_rise, dtype=float)
                    ramp = np.minimum(1.0, (k + 1.0) / L_rise)  # 1/L_rise, ..., 1
                    jitter = 1.0 + self.rise_jitter_std * rng.standard_normal(L_rise)
                    tail[:L_rise] *= ramp * jitter

                x[c:] += tail

            # 4) small Gaussian blur to soften ascents (before noise)
            if self.smooth_sigma_range is not None and self.smooth_radius >= 1:
                sigma_sm = rng.uniform(self.smooth_sigma_range[0], self.smooth_sigma_range[1])
                k = _gauss_kernel1d(sigma_sm, self.smooth_radius)  # normalized kernel
                x = np.convolve(x, k, mode='same')

            # 5) add noise and scale to x_max
            if self.noise_std > 0:
                x += self.noise_std * rng.standard_normal(self.seq_len)
            xmax = np.max(x)
            if xmax > 0:
                x = (self.x_max / xmax) * x
            else:
                x[:] = 0.0

            X_list.append(x.astype(np.float32))
            Y_list.append(y.astype(np.float32))

        X = np.stack(X_list, axis=0)  # (N, T)
        Y = np.stack(Y_list, axis=0)  # (N, T)
        self.X = torch.from_numpy(X).unsqueeze(-1)  # (N, T, 1)
        self.Y = torch.from_numpy(Y).unsqueeze(-1)  # (N, T, 1)

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def plot_predection_example(x, yt, yp=None, yr=None, t=None, title="",
                 legend=True, ax=False, return_ax=False, figsize=(5, 1.8)):
    """
    Plot input x, target yt, optional prediction yp, and optional reference yr.

    Parameters
    ----------
    x, yt : 1D array-like
        Input and target (must have the same length).
    yp : 1D array-like or None, optional
        Optional prediction. If None, no prediction curve is drawn.
    yr : 1D array-like or None, optional
        Optional reference curve (e.g., recursive baseline). Drawn only if not None.
    t : 1D array-like or None
        Optional time axis; if None uses range(len(x)).
    title : str
        Title text (ignored if empty).
    legend : bool
        Show legend if True.
    ax : False | True | matplotlib.axes.Axes
        - False (default): create a new figure/axes and show it (return None).
        - True: create a new figure/axes. If `return_ax` is True, return the Axes
                (no implicit show). Otherwise, show and return None.
        - Axes: draw on the given axes (no implicit show, return None).
    return_ax : bool
        Only meaningful when `ax is True`. If True, return the newly created Axes.
    figsize : (w, h)
        Figure size used when creating new figures.
    """
    # -- coerce to 1D arrays
    xnp  = np.asarray(x).reshape(-1)
    ytnp = np.asarray(yt).reshape(-1)
    assert xnp.shape == ytnp.shape, "x and yt must share the same 1D shape."

    ypnp = None
    if yp is not None:
        ypnp = np.asarray(yp).reshape(-1)
        assert ypnp.shape == xnp.shape, "yp must match the shape of x and yt."

    yrnp = None
    if yr is not None:
        yrnp = np.asarray(yr).reshape(-1)
        assert yrnp.shape == xnp.shape, "yr must match the shape of x and yt."

    # -- time axis
    if t is None:
        t_arr = np.arange(xnp.shape[0], dtype=float)
    else:
        t_arr = np.asarray(t).reshape(-1)
        assert t_arr.shape[0] == xnp.shape[0], "t must match sequence length."

    # -- route by `ax` mode
    if ax is False:
        fig, ax_ = plt.subplots(figsize=figsize)
    elif ax is True:
        fig, ax_ = plt.subplots(figsize=figsize)
    elif isinstance(ax, Axes):
        ax_ = ax
        fig = None
    else:
        raise ValueError("`ax` must be False, True, or a matplotlib Axes instance.")

    # -- draw
    ax_.plot(t_arr, xnp,  linewidth=1.2, color='gray',  label="Input")
    ax_.plot(t_arr, ytnp, linewidth=0.9, color='black', label="Target")
    if ypnp is not None:
        ax_.plot(t_arr, ypnp, linewidth=1.2, color='red',   label="Pred")
    if yrnp is not None:
        ax_.plot(t_arr, yrnp, linewidth=1.0, color='blue', linestyle=":", label="Ref")

    ax_.tick_params(axis='both', labelsize=8)
    if legend:
        ax_.legend(fontsize=8, loc="upper right")
    if title:
        ax_.set_title(title, fontsize=9)

    # -- return/show policy
    if ax is True:
        if return_ax:
            return ax_
        else:
            plt.tight_layout(); plt.show(); return None
    elif ax is False:
        plt.tight_layout(); plt.show(); return None
    else:
        # existing Axes passed in → no implicit show and no return
        return None

def plot_predection_examples(examples, y_ref=None, label_title="", cols=3, cell_size=(3, 1.5)):
    """
    Plot a grid of model prediction examples.

    Parameters
    ----------
    examples : list of (label, x, y, y_pred)
        Each entry contains a panel title label (e.g., epoch, sample) and the
        per-panel sequences (1D array-like, same length).
    y_ref : 1D array-like or None, optional
        Optional reference curve (e.g., recursive filter). Plotted in every panel if not None.
    label_title: str, optional
        Text prefix for each panel title (e.g., "Epoch").
    cols : int, default=3
        Number of columns in the grid.
    cell_size : (float, float), default=(3, 1.5)
        Size of each subplot in inches (width, height).

    Returns
    -------
    fig, axs : matplotlib Figure and 1D array of Axes
    """
    n = len(examples)
    if n == 0:
        fig, ax = plt.subplots(1, 1, figsize=cell_size)
        ax.set_title("No examples to show", fontsize=9)
        ax.axis("off")
        plt.tight_layout()
        return fig, np.array([ax])

    rows = math.ceil(n / cols)
    fig, axs = plt.subplots(
        rows, cols,
        figsize=(cell_size[0] * cols, cell_size[1] * rows),
        sharex=True, sharey=True
    )
    axs = np.atleast_1d(axs).ravel()

    yr_panel = None if y_ref is None else np.asarray(y_ref).reshape(-1)

    for i, (label, x_i, y_i, y_pred_i) in enumerate(examples):
        ax = axs[i]
        xi  = np.asarray(x_i).reshape(-1)
        yi  = np.asarray(y_i).reshape(-1)
        ypi = np.asarray(y_pred_i).reshape(-1)

        title = f"{label_title} {label}".strip()
        # Uses your existing helper; it ignores figsize when an Axes is provided
        plot_predection_example(xi, yi, ypi, yr=yr_panel, ax=ax, legend=False, title=title)

        r, c = divmod(i, cols)
        ax.tick_params(axis='x', labelbottom=(r == rows - 1), labelsize=8)
        ax.tick_params(axis='y', labelleft=(c == 0),   labelsize=8)

    for j in range(n, rows * cols):
        axs[j].axis('off')

    plt.tight_layout()
    return fig, axs
    
def compute_prediction_dataset(model, dataset, indices=None, pred='probs'):
    """
    Compute model predictions for selected dataset items.

    Args:
        model: PyTorch model mapping (1, T, 1) -> (1, T, 1).
        dataset: indexable dataset returning (x, y) shaped (T, 1) or (T,).
        indices: iterable of ints, or None to use all indices in the dataset.
        pred: 'probs' (apply sigmoid) or 'logits' (raw outputs).

    Returns:
        List of tuples (idx, x_np, y_np, y_pred_np) with numpy arrays.
    """
    if pred not in ('probs', 'logits'):
        raise ValueError("pred must be 'probs' or 'logits'.")

    # Use all indices if none provided
    if indices is None:
        indices = range(len(dataset))

    results = []
    model.eval()

    # Place tensors on the same device as the model
    device = next(model.parameters()).device

    with torch.no_grad():
        for idx in indices:
            x, y = dataset[idx]  # x: (T,1) or (T,), y: (T,1) or (T,)

            # Ensure (T,1) shape
            if x.dim() == 1: x = x.unsqueeze(-1)
            if y.dim() == 1: y = y.unsqueeze(-1)

            # Add batch dim and move to device
            x_b = x.unsqueeze(0).to(device)  # (1, T, 1)

            # Forward pass
            logits = model(x_b)              # (1, T, 1)
            y_pred = torch.sigmoid(logits) if pred == 'probs' else logits

            # Move to CPU and drop batch/feature dims
            x_np  = x[:, 0].detach().numpy()
            y_np  = y[:, 0].detach().numpy()
            yp_np = y_pred[0, :, 0].detach().numpy()

            results.append((int(idx), x_np, y_np, yp_np))

    return results

def eval_PRF(B_ref, B_pred, tolerance=0):
    """
    Compute precision (P), recall (R), and F-measure (F) for event detection
    with a temporal tolerance and greedy one-to-one matching.

    Parameters
    ----------
    B_ref : list or array-like of int
        Reference events (indices).
    B_pred : list or array-like of int
        Predicted events (indices).
    tolerance : int, optional (default = 0)
        Maximum allowed deviation |b_pred - b_ref| to count a match.

    Returns
    -------
    P : float
        Precision = TP / (TP + FP)
    R : float
        Recall    = TP / (TP + FN)
    F : float
        F-measure = 2 * P * R / (P + R)
    cond_ref : bool
        True iff reference events satisfy the separation condition
        |b_ref[k+1] - b_ref[k]| > 2 * tolerance  (or vacuously true for <=1 ref)
    cond_pred : bool
        True iff predicted events satisfy the separation condition
        |b_pred[k+1] - b_pred[k]| > 2 * tolerance  (or vacuously true for <=1 pred)
    """
    # Convert to sorted numpy arrays (integers)
    B_ref = np.sort(np.asarray(B_ref, dtype=int))
    B_pred = np.sort(np.asarray(B_pred, dtype=int))

    # Separation conditions (non-overlapping tolerance windows)
    cond_ref = np.all(np.diff(B_ref) > 2 * tolerance) if B_ref.size > 1 else True
    cond_pred = np.all(np.diff(B_pred) > 2 * tolerance) if B_pred.size > 1 else True

    # Track matches for greedy one-to-one assignment
    matched_ref = np.zeros(B_ref.size, dtype=bool)
    matched_pred = np.zeros(B_pred.size, dtype=bool)

    # Greedy: iterate predictions, match nearest unmatched reference within tolerance
    for i, b_pred in enumerate(B_pred):
        diffs = np.abs(B_ref - b_pred)
        candidates = np.where((diffs <= tolerance) & (~matched_ref))[0]
        if candidates.size:
            j = candidates[np.argmin(diffs[candidates])]
            matched_ref[j] = True
            matched_pred[i] = True

    TP = int(matched_pred.sum())
    FP = int((~matched_pred).sum())
    FN = int((~matched_ref).sum())

    # Metrics
    P = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    R = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    F = (2 * P * R / (P + R)) if (P + R) > 0 else 0.0

    return P, R, F, cond_ref, cond_pred

def mean_PRF(results):
    """
    Return mean P, R, F and AND of cond_ref, cond_pred.
    """
    P_vals, R_vals, F_vals = [], [], []
    cond_ref_all, cond_pred_all = True, True

    for r in results:
        _, P, R, F, cond_ref, cond_pred, *_ = r
        P_vals.append(P)
        R_vals.append(R)
        F_vals.append(F)
        cond_ref_all = cond_ref_all and cond_ref
        cond_pred_all = cond_pred_all and cond_pred

    P_mean = float(np.mean(P_vals)) if P_vals else 0.0
    R_mean = float(np.mean(R_vals)) if R_vals else 0.0
    F_mean = float(np.mean(F_vals)) if F_vals else 0.0

    return P_mean, R_mean, F_mean, cond_ref_all, cond_pred_all

def plot_peaks_stem(
    y_ref=None, peaks_ref=None, props_ref=None,
    y_pred=None, peaks_pred=None, props_pred=None,
    val_min=None, tolerance=None, ax=None, title=None, show_legend=True
):
    """Thin curves; stems width=1; black ref dots slightly larger than red est dots."""
    assert (y_ref is not None) or (y_pred is not None), "Provide at least y_ref or y_pred"
    fig, ax = (None, ax) if ax is not None else plt.subplots(figsize=(6, 1.5))

    def _one(y, peaks, props, color, label, marker_size, stem_lw=1.0):
        if y is None: 
            return np.array([], dtype=int)
        y = np.asarray(y).reshape(-1)
        t = np.arange(len(y))
        ax.plot(t, y, color=color, lw=0.7, alpha=0.9, label=label)  # thin curve
        if peaks is None: 
            return np.array([], dtype=int)
        peaks = np.asarray(peaks, dtype=int)
        if peaks.size:
            h = props.get("peak_heights") if (props and "peak_heights" in props) else y[peaks]
            m, s, _ = ax.stem(peaks, h, linefmt=color, markerfmt=color+"o", basefmt=" ")
            m.set_markersize(marker_size)   # ref > est
            s.set_linewidth(stem_lw)        # stems as before (thicker than curve)
        return peaks

    if val_min is not None:
        ax.axhline(val_min, ls="--", lw=1, color="gray", alpha=0.8)

    # ref: black, bigger dot; est: red, smaller dot
    ref_peaks = _one(y_ref, peaks_ref, props_ref, color="k", label="Ref", marker_size=3.5, stem_lw=1.0)
    _one(y_pred, peaks_pred, props_pred, color="r", label="Est", marker_size=3, stem_lw=1.0)

    # tolerance guides around reference peaks
    if (tolerance is not None) and (tolerance > 0) and ref_peaks.size:
        ymin, ymax = ax.get_ylim()
        for p in ref_peaks:
            ax.vlines([p - tolerance, p + tolerance], ymin, ymax, colors="k", linestyles=":", linewidth=0.8, alpha=0.75)

    ax.set_title(title or "Peaks", fontsize=9)
    ax.set_xlabel("Index", fontsize=8)
    ax.set_ylabel("Prob", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.3)
    if show_legend:
        ax.legend(loc="lower right", fontsize=8)
    if fig is not None:
        fig.tight_layout()
    return ax
    
def eval_peaks_dataset(examples, tolerance=0, height=0.5, distance=None,
                       plot=False, print_summary=True):
    """
    Evaluate peak detection across a set of examples.

    Args:
        examples: list of tuples (idx, x_np, y_np, yp_np) from compute_prediction_dataset
        tolerance (int or float): match tolerance (in samples); internally cast to int
        height (float): minimum peak height for both reference and prediction
        distance (int or None): minimum distance between detected peaks.
            If None → distance = 2*int(tolerance) + 1
        plot (bool): if True, call plot_peaks_stem for each example
        print_summary (bool): if True, print per-example summary

    Returns:
        results: list of tuples per example:
          (idx, P, R, F, cond_ref, cond_pred, peaks_ref, props_ref, peaks_pred, props_pred)
    """
    results = []

    tol = int(tolerance)
    if tol < 0:
        raise ValueError("tolerance must be >= 0")

    # default for distance if not provided
    if distance is None:
        distance = 2 * tol + 1
    distance = int(distance)
    if distance < 1:
        distance = 1  # scipy.signal.find_peaks requires distance >= 1

    for (idx, x_np, y_np, yp_np) in examples:
        y_ref = np.asarray(y_np).reshape(-1)
        y_pred = np.asarray(yp_np).reshape(-1)

        peaks_ref, props_ref = find_peaks(
            y_ref, height=height, distance=distance, plateau_size=True
        )
        peaks_pred, props_pred = find_peaks(
            y_pred, height=height, distance=distance, plateau_size=True
        )

        P, R, F, cond_ref, cond_pred = eval_PRF(peaks_ref, peaks_pred, tolerance=tol)

        if print_summary:
            print(f"[idx={idx}] P={P:.3f} | R={R:.3f} | F={F:.3f} "
                  f"| cond_ref={cond_ref} | cond_pred={cond_pred}")

        if plot:
            plot_peaks_stem(
                y_ref, peaks_ref, props_ref,
                y_pred, peaks_pred, props_pred,
                val_min=height, tolerance=tol,
                title=f"[idx={idx}] P={P:.3f} | R={R:.3f} | F={F:.3f} "
                      f"| cond_ref={cond_ref} | cond_pred={cond_pred}"
            )
            plt.show()

        results.append(
            (idx, P, R, F, cond_ref, cond_pred, peaks_ref, props_ref, peaks_pred, props_pred)
        )

    return results
 
# ================================================
# Exercise 1: exercise_recursive_impulse()
# ================================================
# 1) Echo: y[n] = x[n] + gain * y[n-delay]
def echo(x, delay=3, gain=0.6, length=None):
    T = length if length is not None else len(x)
    y = np.zeros(T, dtype=float)
    for n in range(T):
        x_n = x[n] if n < len(x) else 0.0
        y[n] = x_n + (gain * y[n - delay] if n >= delay else 0.0)
    return y

# 2) Smoothing (leaky average): y[n] = s*x[n] + (1-s)*y[n-1]
def smoothing(x, s=0.5, length=None):
    T = length if length is not None else len(x)
    y = np.zeros(T, dtype=float)
    for n in range(T):
        x_n = x[n] if n < len(x) else 0.0
        y_prev = y[n-1] if n > 0 else 0.0
        y[n] = s * x_n + (1 - s) * y_prev
    return y

# 3) Change detector (recursive): y[n] = r*y[n-1] + k*(x[n]-x[n-1]), y[0]=0
def change_detector_recursive(x, r=0.6, k=1.0, length=None):
    T = length if length is not None else len(x)
    y = np.zeros(T, dtype=float)
    for n in range(1, T):
        x_n   = x[n]   if n   < len(x) else 0.0
        x_prev= x[n-1] if n-1 < len(x) else 0.0
        y[n] = r * y[n-1] + k * (x_n - x_prev)
    return y

# 4) Running sum with decay: y[n] = r*y[n-1] + x[n]
def running_sum_decay(x, r=0.8, length=None):
    T = length if length is not None else len(x)
    y = np.zeros(T, dtype=float)
    for n in range(T):
        x_n = x[n] if n < len(x) else 0.0
        y_prev = y[n-1] if n > 0 else 0.0
        y[n] = r * y_prev + x_n
    return y
    
def exercise_recursive_impulse():
    """Exercise 1: Impulse Responses of Recursive Filters

    Notebook: PCPT_09_recursion.ipynb
    """
    # === Part 1: Impulse responses for each filter type ===
    print("=== Impulse responses for Echo, Smoothing, Change Detector, and Running Sum with Decay ===")

    # Impulse input
    x_imp = np.array([1, 0, 0, 0, 0, 0, 0], float)
    length_out = 12  # show a bit beyond input length

    # Compute impulse responses using the provided functions
    h_echo   = echo(x_imp, delay=3, gain=0.6, length=length_out)
    h_smooth = smoothing(x_imp, s=0.5, length=length_out)
    h_change = change_detector_recursive(x_imp, r=0.6, k=1.0, length=length_out)
    h_sum    = running_sum_decay(x_imp, r=0.8, length=length_out)

    # Plot responses
    fig, axs = plt.subplots(2, 2, figsize=(6, 3.5))

    plot_io(x_imp, h_echo,   title="Impulse Response: Echo",
            ax=axs[0, 0], xlim=(-0.5, length_out - 0.5), offset=0.1)

    plot_io(x_imp, h_smooth, title="Impulse Response: Smoothing",
            ax=axs[0, 1], xlim=(-0.5, length_out - 0.5), offset=0.1)

    plot_io(x_imp, h_change, title="Impulse Response: Change Detector",
            ax=axs[1, 0], xlim=(-0.5, length_out - 0.5), offset=0.1)

    plot_io(x_imp, h_sum,    title="Impulse Response: Running Sum with Decay",
            ax=axs[1, 1], xlim=(-0.5, length_out - 0.5), offset=0.1)

    plt.tight_layout()
    plt.show()


    # === Part 2: Change detector with different parameter settings ===
    print("\n=== Change detector impulse responses for different (r, k) settings ===")

    # Impulse input for change detector tests
    x_imp = np.zeros(length_out)
    x_imp[0] = 1.0

    # Four parameter settings for (r, k)
    params = [
        (0.6, 1.0),  # original
        (0.9, 1.0),  # slower decay
        (1.2, 1.0),  # increase
        (0.6, 2.5)   # stronger initial change
    ]

    # Compute and plot responses for each setting
    fig, axs = plt.subplots(2, 2, figsize=(6, 3.5))
    for ax, (r_val, k_val) in zip(axs.flat, params):
        h_change = change_detector_recursive(x_imp, r=r_val, k=k_val, length=length_out)
        plot_io(x_imp, h_change, f"Change Detector: r={r_val}, k={k_val}", ax=ax, offset=0.1)
        
        # Set grid spacing to 1 for all subplots
        ax.set_xticks(np.arange(0, length_out, 1))
        ax.set_yticks(np.arange(int(h_change.min()) - 1, int(h_change.max()) + 2, 1))
        ax.grid(True, which='both', linewidth=0.5)

    plt.tight_layout()
    plt.show()
 
# ================================================
# Exercise 2: exercise_weighting_pos()
# ================================================
def make_bce_loss_from_dataset(train_ds):
    """
    Computes a simple pos_weight for BCE to counter class imbalance
    (spikes are rare). Targets should be 0/1 or in [0,1].
    """
    with torch.no_grad():
        pos_frac = (train_ds.Y > 0.5).float().mean().item()  # works for binary masks
    pos_weight = torch.tensor([(1.0 - pos_frac) / max(pos_frac, 1e-6)], dtype=torch.float32)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
def exercise_weighting_pos():
    """Exercise 2:  Class-Imbalance Weighting

    Notebook: PCPT_09_recursion.ipynb
    """
    settings = [
        {"sigma": None, "min_gap": 10, "peaks": (3, 6), "seed": 0},    
        {"sigma": 1.0, "min_gap": 10, "peaks": (3, 6), "seed": 0},
        {"sigma": 2.0, "min_gap": 10, "peaks": (3, 6), "seed": 0},
        {"sigma": 2.0, "min_gap": 10, "peaks": (3, 6), "seed": 1},
        {"sigma": 1.0, "min_gap":  5, "peaks": (3, 8), "seed": 0},   
        {"sigma": 1.0, "min_gap":  5, "peaks": (5, 10), "seed": 0},  
        {"sigma": 1.0, "min_gap":  5, "peaks": (8, 12), "seed": 0}, 
        {"sigma": 1.0, "min_gap":  5, "peaks": (8, 12), "seed": 1},
    ]
    eps = 1e-5    

    print("Using:")
    print("ds = NoveltyCurveDataset(n_samples=200, seq_len=128, sigma_target, min_gap, n_peaks_range, seed)")
    print(f"epsilon = {eps}\n")

                
    print("sigma  | min_gap | n_peaks_range | seed | pos_frac |    pw_own | pw_from_loss | abs_diff ")
    print("----------------------------------------------------------------------------------------")

    for cfg in settings:
        ds = NoveltyCurveDataset(
            n_samples=200,
            seq_len=128,
            sigma_target=cfg["sigma"],
            min_gap=cfg["min_gap"],
            n_peaks_range=cfg["peaks"],
            seed=cfg["seed"]
        )

        with torch.no_grad():
            pos_frac = (ds.Y > 0.5).float().mean().item()

        pw_own = (1.0 - pos_frac) / max(pos_frac, eps)
        loss_fn = make_bce_loss_from_dataset(ds)
        pw_from_loss = loss_fn.pos_weight.item() if loss_fn.pos_weight.numel() == 1 else float(loss_fn.pos_weight.mean())

        sigma_str = "None" if cfg["sigma"] is None else f"{cfg['sigma']:.1f}"

        print(f"{sigma_str:6} | {cfg['min_gap']:7d} | {str(cfg['peaks']):13s} | "
              f"{cfg['seed']:4d} | {pos_frac:8.4f} | {pw_own:9.2f} | {pw_from_loss:12.2f} | "
              f"{abs(pw_own - pw_from_loss):8.2e}")

    print("\nNote: Differences between pw_own and pw_from_loss are due to different epsilon values used to avoid division by zero.")    
    
 
# ================================================
# Exercise 3: exercise_training_prf()
# ================================================
# ------------------------------------------------------------
# 1) Model: plain RNN (tanh) -> per-frame logits
# ------------------------------------------------------------
class SimpleTanhRNN(nn.Module):
    """
    Vanilla tanh RNN for per-frame logits.

    - Model capacity: hidden_size = H controls temporal capacity ("memory slots").
    - Initialization: random weights; no built-in filter recursion (learn from data).
    - Outputs: Linear readout -> 1 logit per frame (pass to BCEWithLogitsLoss; no sigmoid here).
    - Context direction: bidirectional=True uses past and future context (non-causal, 2H hidden).
    - Shapes (batch_first=True):
        x      (B, T, 1)
        h_seq  (B, T, H) or (B, T, 2H) if bidirectional
        logits (B, T, 1)
    """
    def __init__(self, hidden_size=8, bidirectional=False, batch_first=True):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=1,
            nonlinearity="tanh",
            bias=True,
            batch_first=batch_first,
            bidirectional=bidirectional,
        )
        d = 2 if bidirectional else 1
        self.readout = nn.Linear(d * hidden_size, 1, bias=True)

    def forward(self, x, h0=None):
        h_seq, _ = self.rnn(x, h0)
        logits = self.readout(h_seq)
        return logits

def evaluate_model_loader(model, loader, loss_fn, use_sigmoid=False):
    """
    Simple evaluation loop.
    Run the model on all batches in a DataLoader and return the average loss.

    Args:
        model: Maps (B, T, 1) to (B, T, 1).
        loader: DataLoader yielding batches (xb, yb).
        loss_fn: Loss function, e.g., nn.MSELoss() or nn.BCELoss().
        use_sigmoid (bool): If True, apply sigmoid to outputs before loss.

    Notes:
        - use_sigmoid=False: Pass raw model outputs to loss_fn  
          (use for MSELoss or BCEWithLogitsLoss).
        - use_sigmoid=True: Apply sigmoid to the outputs before loss_fn  
          (use for BCELoss, which expects probabilities in [0, 1]).
    """
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            yb_pred = model(xb)
            if use_sigmoid:
                yb_pred = torch.sigmoid(yb_pred)
            loss = loss_fn(yb_pred, yb)        # Compute batch loss
            total += loss.item() * xb.size(0)  # Accumulate summed loss
            n += xb.size(0)                    # Count number of samples
    return total / max(n, 1)                   # Average loss over all samples
  
# Consistent helper name and docstring; returns only (P_mean, R_mean, F_mean)
def evaluate_model_dataset(model, ds, tolerance=2, height=0.4, distance=None):
    """
    Compute macro-averaged precision (P_mean), recall (R_mean), and F-measure (F_mean)
    for a model on a dataset using peak picking.

    Args:
        model: Trained PyTorch model.
        ds:    Dataset with (x, y) pairs.
        tolerance (int): Matching tolerance in samples (± frames).
        height (float):  Minimum peak height for detection.
        distance (int | None): Minimum spacing between peaks. If None, use 2*tolerance + 1
                               to encourage one-to-one matches.

    Returns:
        (P_mean, R_mean, F_mean): floats with macro-averaged PRF over samples.
    """
    if distance is None:
        distance = 2 * tolerance + 1

    # Run the model on the whole dataset and collect probabilities
    pred_ds = compute_prediction_dataset(model, ds, indices=None, pred='probs')

    # Peak picking and PRF evaluation
    eval_ds = eval_peaks_dataset(
        pred_ds,
        tolerance=tolerance,
        height=height,
        distance=None,
        plot=False,
        print_summary=False
    )

    P_mean, R_mean, F_mean, _, _ = mean_PRF(eval_ds)
    return P_mean, R_mean, F_mean 
    
# ------------------------------------------------------------
# Train loop (simple, stable defaults)
# ------------------------------------------------------------
def train_SimpleTanhRNN_PRF(train_loader, val_loader, train_ds, val_ds,
                        hidden_size=8, bidirectional=False,
                        lr=0.01, epochs=20, clip=1.0, print_every=10,
                        snapshot_idx=0, weight_balance=True,
                        tolerance=2, height=0.4, distance=None):
    """
    Train SimpleTanhRNN with BCEWithLogitsLoss and capture prediction snapshots.

    Notes:
      - hidden_size controls model capacity
      - bidirectional=True adds past+future context (non-causal)
      - gradient clipping keeps training stable
      - snapshots show how predictions evolve on one fixed training sample
      - weight_balance=True uses class weighting to counter rare positives
    """
    model = SimpleTanhRNN(hidden_size=hidden_size, bidirectional=bidirectional)

    # choose loss: with or without class weighting
    if weight_balance:
        loss_fn = make_bce_loss_from_dataset(train_ds)
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # pick one fixed sample for snapshots
    x_fix, y_fix = train_ds[snapshot_idx]
    if x_fix.dim() == 1: x_fix = x_fix.unsqueeze(-1)
    if y_fix.dim() == 1: y_fix = y_fix.unsqueeze(-1)
    x_fix_np, y_fix_np = x_fix[:, 0].numpy(), y_fix[:, 0].numpy()

    snapshots = []  # list of (epoch, x_np, y_np, y_pred_np)

    for ep in range(1, epochs + 1):
        model.train()
        total, count = 0.0, 0
        for xb, yb in train_loader:
            opt.zero_grad()
            logits = model(xb)           # (B, T, 1)
            loss = loss_fn(logits, yb)   # BCE on logits
            loss.backward()
            if clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            total += loss.item() * xb.size(0)
            count += xb.size(0)

        train_loss = total / max(count, 1)
        val_loss   = evaluate_model_loader(model, val_loader, loss_fn, use_sigmoid=False)

        # --- NEW: compute PRF on train_ds and val_ds ---
        train_P, train_R, train_F = evaluate_model_dataset(
            model, train_ds, tolerance=tolerance, height=height, distance=distance
        )
        val_P, val_R, val_F = evaluate_model_dataset(
            model, val_ds, tolerance=tolerance, height=height, distance=distance
        )

        if (ep % print_every == 0) or ep == 1 or ep == epochs:            
            print(f"Ep {ep:2d} | T-L: {train_loss:.3f} | V-L: {val_loss:.3f} "
                  f"| T-PRF: {train_P:.3f}/{train_R:.3f}/{train_F:.3f} "
                  f"| V-PRF: {val_P:.3f}/{val_R:.3f}/{val_F:.3f}")
            model.eval()
            with torch.no_grad():
                y_pred = torch.sigmoid(model(x_fix.unsqueeze(0)))[0, :, 0].numpy()
            snapshots.append((ep, x_fix_np, y_fix_np, y_pred.copy()))

    return model, snapshots
    
def exercise_training_prf():
    """Exercise 3: Exploring Model Training with PRF Reporting

    Notebook: PCPT_09_recursion.ipynb
    """
    # Build datasets and DataLoaders 
    train_ds = NoveltyCurveDataset(n_samples=200, seq_len=128, sigma_target=1.0, seed=0)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_ds = NoveltyCurveDataset(n_samples=100, seq_len=128, sigma_target=1.0, seed=1)
    val_loader = DataLoader(val_ds,   batch_size=32)

    # Train and collect snapshots on one fixed example
    sample_idx = 2
    tolerance = 2
    height = 0.4

    # Train and collect snapshots on one fixed example
    print("=============================================")
    print("Training on train_ds and validation on val_ds")
    print(f"PRF settings: tolerance={tolerance}, height={height}")
    print("=============================================")

    print("train_ds = NoveltyCurveDataset(n_samples=200, seq_len=128, sigma_target=1.0, seed=0)")
    print("val_ds  =  NoveltyCurveDataset(n_samples=100, seq_len=128, sigma_target=1.0, seed=1)")

    sample_idx = 2
    torch.manual_seed(0)
    model, snapshots = train_SimpleTanhRNN_PRF(
        train_loader, val_loader, train_ds, val_ds,
        hidden_size=8,
        bidirectional=True,
        lr=0.01,
        epochs=15,
        clip=1.0,
        print_every=1,
        snapshot_idx=sample_idx,
        weight_balance=True,
        # evaluation parameters (used for PRF during training)
        tolerance=tolerance,
        height=height,
        distance=None  # will default internally to 2*tolerance+1 if None
    )

    print("\nPlot of selected examples:\n")

    pred_ds = compute_prediction_dataset(model, val_ds, indices=[2, 6], pred='probs')
    eval_ds = eval_peaks_dataset(
        pred_ds, tolerance=tolerance, height=height, distance=None,
        plot=True, print_summary=False
    )    
    
    # Build datasets and DataLoaders 
    train_ds_mod = NoveltyCurveDataset(n_samples=200, seq_len=128, sigma_target=1.0,
        min_gap=6, n_peaks_range=(5, 8), noise_std=0.06, rise_len_rng=(2, 10), seed=0)
    train_loader_mod = DataLoader(train_ds_mod, batch_size=32)

    val_ds_mod = NoveltyCurveDataset(n_samples=200, seq_len=128, sigma_target=1.0,
        min_gap=6, n_peaks_range=(5, 8), noise_std=0.06, rise_len_rng=(2, 10), seed=1)
    val_loader_mod = DataLoader(val_ds_mod, batch_size=32)

    # Train and collect snapshots on one fixed example
    sample_idx = 2
    tolerance = 1
    height = 0.5

    print("=============================================================")
    print("Training on train_ds_mod and validation on val_ds_mod        ")
    print(f"PRF settings: tolerance={tolerance}, height={height}")
    print("=============================================================")

    print("train_ds_mod = NoveltyCurveDataset(n_samples=200, seq_len=128, sigma_target=1.0,")
    print("    min_gap=6, n_peaks_range=(5, 8), noise_std=0.06, rise_len_rng=(2, 10), seed=0)")
    print("val_ds_mod = NoveltyCurveDataset(n_samples=100, seq_len=128, sigma_target=1.0,")
    print("    min_gap=6, n_peaks_range=(5, 8), noise_std=0.06, rise_len_rng=(2, 10), seed=1)")

    torch.manual_seed(0)
    model, snapshots = train_SimpleTanhRNN_PRF(
        train_loader_mod, val_loader_mod, train_ds_mod, val_ds_mod,
        hidden_size=8,
        bidirectional=True,
        lr=0.01,
        epochs=15,
        clip=1.0,
        print_every=1,
        snapshot_idx=sample_idx,
        weight_balance=True,
        # evaluation parameters (used for PRF during training)
        tolerance=tolerance,
        height=height,
        distance=None  # will default internally to 2*tolerance+1 if None
    )

    print("\nPlot of selected examples:\n")

    pred_ds = compute_prediction_dataset(model, val_ds_mod, indices=[4, 5], pred='probs')
    eval_ds = eval_peaks_dataset(
        pred_ds, tolerance=tolerance, height=height, distance=None,
        plot=True, print_summary=False
    )
        