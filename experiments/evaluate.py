#!/usr/bin/env python3
"""Shared evaluation utilities for gnss_gpu experiments.

Functions
---------
compute_metrics(estimated, ground_truth)
    Returns dict with rms_2d, rms_3d, mean, std, p50, p67, p95, max.

plot_cdf(errors_dict, output_path)
    Plot CDF curves for multiple methods.

plot_error_timeline(times, errors_dict, output_path)
    Plot error over time.

save_results(results_dict, output_path)
    Save to CSV.

print_comparison_table(results_dict)
    Print formatted table.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from gnss_gpu.range_model import geometric_ranges_sagnac, rotate_satellites_sagnac


# ---------------------------------------------------------------------------
# WGS84 constants
# ---------------------------------------------------------------------------
_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = 2.0 * _WGS84_F - _WGS84_F ** 2


def ecef_to_lla(x: float, y: float, z: float):
    """Convert ECEF to geodetic (lat_rad, lon_rad, alt_m)."""
    b = _WGS84_A * (1.0 - _WGS84_F)
    ep2 = (_WGS84_A ** 2 - b ** 2) / (b ** 2)
    p = math.sqrt(x ** 2 + y ** 2)
    theta = math.atan2(z * _WGS84_A, p * b)
    lat = math.atan2(
        z + ep2 * b * math.sin(theta) ** 3,
        p - _WGS84_E2 * _WGS84_A * math.cos(theta) ** 3,
    )
    lon = math.atan2(y, x)
    sin_lat = math.sin(lat)
    N = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat ** 2)
    if abs(math.cos(lat)) > 1e-10:
        alt = p / math.cos(lat) - N
    else:
        alt = abs(z) - b
    return lat, lon, alt


def lla_to_ecef(lat_rad: float, lon_rad: float, alt: float) -> np.ndarray:
    """Convert geodetic (lat_rad, lon_rad, alt_m) to ECEF."""
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    N = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat ** 2)
    x = (N + alt) * cos_lat * math.cos(lon_rad)
    y = (N + alt) * cos_lat * math.sin(lon_rad)
    z = (N * (1.0 - _WGS84_E2) + alt) * sin_lat
    return np.array([x, y, z])


def ecef_errors_2d_3d(
    estimated: np.ndarray,
    ground_truth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 2D (horizontal) and 3D errors in ENU frame.

    Parameters
    ----------
    estimated : ndarray, shape (N, 3)
        Estimated ECEF positions [m].
    ground_truth : ndarray, shape (N, 3) or (3,)
        Ground-truth ECEF positions [m].  If (3,), broadcast to all epochs.

    Returns
    -------
    errors_2d : ndarray, shape (N,)
        Horizontal (ENU east+north) errors [m].
    errors_3d : ndarray, shape (N,)
        3D errors [m].
    """
    est = np.asarray(estimated, dtype=np.float64).reshape(-1, 3)
    gt = np.asarray(ground_truth, dtype=np.float64)
    if gt.ndim == 1:
        gt = np.tile(gt, (len(est), 1))
    gt = gt.reshape(-1, 3)

    n = len(est)
    errors_2d = np.zeros(n)
    errors_3d = np.zeros(n)

    for i in range(n):
        ref = gt[i]
        lat, lon, _ = ecef_to_lla(ref[0], ref[1], ref[2])
        sin_lat = math.sin(lat)
        cos_lat = math.cos(lat)
        sin_lon = math.sin(lon)
        cos_lon = math.cos(lon)

        # ECEF to ENU rotation
        R = np.array([
            [-sin_lon, cos_lon, 0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ])

        diff = est[i] - ref
        enu = R @ diff
        errors_2d[i] = math.sqrt(enu[0] ** 2 + enu[1] ** 2)
        errors_3d[i] = math.sqrt(enu[0] ** 2 + enu[1] ** 2 + enu[2] ** 2)

    return errors_2d, errors_3d


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def compute_metrics(
    estimated: np.ndarray,
    ground_truth: np.ndarray,
) -> dict:
    """Compute positioning accuracy metrics.

    Parameters
    ----------
    estimated : array_like, shape (N, 3) or (N, 4)
        Estimated ECEF positions [m].  If shape (N, 4), first 3 columns used.
    ground_truth : array_like, shape (N, 3) or (3,)
        Ground-truth ECEF positions [m].

    Returns
    -------
    metrics : dict
        Keys: rms_2d, rms_3d, mean_2d, mean_3d, std_2d, p50, p67, p95, max_2d,
              n_epochs, errors_2d, errors_3d.
    """
    est = np.asarray(estimated, dtype=np.float64)
    if est.ndim == 2 and est.shape[1] >= 4:
        est = est[:, :3]
    est = est.reshape(-1, 3)

    errors_2d, errors_3d = ecef_errors_2d_3d(est, ground_truth)

    return {
        "rms_2d": float(np.sqrt(np.mean(errors_2d ** 2))),
        "rms_3d": float(np.sqrt(np.mean(errors_3d ** 2))),
        "mean_2d": float(np.mean(errors_2d)),
        "mean_3d": float(np.mean(errors_3d)),
        "std_2d": float(np.std(errors_2d)),
        "p50": float(np.percentile(errors_2d, 50)),
        "p67": float(np.percentile(errors_2d, 67)),
        "p95": float(np.percentile(errors_2d, 95)),
        "max_2d": float(np.max(errors_2d)),
        "n_epochs": int(len(errors_2d)),
        "errors_2d": errors_2d,
        "errors_3d": errors_3d,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_cdf(
    errors_dict: Dict[str, np.ndarray],
    output_path: str | Path,
    title: str = "CDF of 2D Positioning Error",
    xlabel: str = "2D Error [m]",
) -> None:
    """Plot CDF curves for multiple methods.

    Parameters
    ----------
    errors_dict : dict
        Keys are method labels, values are 1-D error arrays.
    output_path : str or Path
        Path to save the figure (PNG).
    title : str
        Figure title.
    xlabel : str
        X-axis label.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[evaluate] matplotlib not available, skipping CDF plot: {output_path}")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2"]

    for idx, (label, errors) in enumerate(errors_dict.items()):
        errs = np.sort(np.asarray(errors, dtype=np.float64))
        cdf = np.arange(1, len(errs) + 1) / len(errs)
        ls = linestyles[idx % len(linestyles)]
        col = colors[idx % len(colors)]
        ax.plot(errs, cdf * 100.0, linestyle=ls, color=col,
                linewidth=1.8, label=label)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Cumulative Probability [%]", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 100)

    # Mark p50, p95 reference lines
    ax.axhline(50, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.axhline(95, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.text(ax.get_xlim()[1] * 0.99, 51, "50%", ha="right", fontsize=8, color="gray")
    ax.text(ax.get_xlim()[1] * 0.99, 96, "95%", ha="right", fontsize=8, color="gray")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] CDF plot saved: {output_path}")


def plot_error_timeline(
    times: np.ndarray,
    errors_dict: Dict[str, np.ndarray],
    output_path: str | Path,
    title: str = "Positioning Error Over Time",
    ylabel: str = "2D Error [m]",
) -> None:
    """Plot error over time for multiple methods.

    Parameters
    ----------
    times : array_like, shape (N,)
        Time vector (e.g., epoch index or GPS seconds).
    errors_dict : dict
        Keys are method labels, values are 1-D error arrays.
    output_path : str or Path
        Path to save the figure (PNG).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[evaluate] matplotlib not available, skipping timeline plot: {output_path}")
        return

    fig, ax = plt.subplots(figsize=(10, 4))

    linestyles = ["-", "--", "-.", ":"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    t = np.asarray(times, dtype=np.float64)

    for idx, (label, errors) in enumerate(errors_dict.items()):
        errs = np.asarray(errors, dtype=np.float64)
        n = min(len(t), len(errs))
        ls = linestyles[idx % len(linestyles)]
        col = colors[idx % len(colors)]
        ax.plot(t[:n], errs[:n], linestyle=ls, color=col,
                linewidth=1.5, label=label, alpha=0.85)

    ax.set_xlabel("Time [epoch]", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(t[0], t[-1])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] Timeline plot saved: {output_path}")


def plot_pareto(
    times_dict: Dict[str, float],
    accuracy_dict: Dict[str, float],
    output_path: str | Path,
    title: str = "Accuracy vs Computation Time (Pareto Frontier)",
    xlabel: str = "Computation Time per Epoch [ms]",
    ylabel: str = "Mean 2D Error [m]",
) -> None:
    """Plot Pareto frontier (accuracy vs time).

    Parameters
    ----------
    times_dict : dict
        Keys are method labels, values are computation time [ms].
    accuracy_dict : dict
        Keys are method labels, values are accuracy metric (lower is better).
    output_path : str or Path
        Path to save the figure (PNG).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[evaluate] matplotlib not available, skipping Pareto plot: {output_path}")
        return

    labels = list(times_dict.keys())
    times = [times_dict[k] for k in labels]
    accs = [accuracy_dict[k] for k in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(times, accs, s=80, zorder=5)

    for label, t, a in zip(labels, times, accs):
        ax.annotate(label, (t, a), textcoords="offset points",
                    xytext=(6, 4), fontsize=9)

    # Draw Pareto frontier
    pts = sorted(zip(times, accs))
    pareto_x, pareto_y = [], []
    min_acc = float("inf")
    for tx, ax_ in pts:
        if ax_ < min_acc:
            pareto_x.append(tx)
            pareto_y.append(ax_)
            min_acc = ax_
    ax.plot(pareto_x, pareto_y, "r--", linewidth=1.5,
            label="Pareto frontier", alpha=0.7)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] Pareto plot saved: {output_path}")


# ---------------------------------------------------------------------------
# I/O utilities
# ---------------------------------------------------------------------------

def save_results(results_dict: dict, output_path: str | Path) -> None:
    """Save results dictionary to CSV.

    Parameters
    ----------
    results_dict : dict
        Keys are column names, values are scalars or 1-D arrays.
        All array-valued entries must have the same length; scalar entries
        are broadcast.
    output_path : str or Path
        Destination CSV file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Separate scalar vs array fields
    scalar_fields = {}
    array_fields = {}
    for k, v in results_dict.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1:
            array_fields[k] = v
        elif isinstance(v, (list, tuple)):
            array_fields[k] = np.asarray(v)
        else:
            scalar_fields[k] = v

    if not array_fields:
        # All scalars: write single row
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results_dict.keys()))
            writer.writeheader()
            writer.writerow({k: _fmt(v) for k, v in results_dict.items()})
    else:
        # Determine row count from first array
        n = len(next(iter(array_fields.values())))
        fieldnames = list(scalar_fields.keys()) + list(array_fields.keys())
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(n):
                row = {k: _fmt(scalar_fields[k]) for k in scalar_fields}
                for k, arr in array_fields.items():
                    row[k] = _fmt(arr[i] if i < len(arr) else "")
                writer.writerow(row)

    print(f"[evaluate] Results saved: {output_path}")


def _fmt(v) -> str:
    """Format a value for CSV output."""
    if isinstance(v, float):
        return f"{v:.6f}"
    if isinstance(v, (np.floating,)):
        return f"{float(v):.6f}"
    if isinstance(v, (np.integer,)):
        return str(int(v))
    return str(v)


def print_comparison_table(results_dict: Dict[str, dict]) -> None:
    """Print a formatted comparison table.

    Parameters
    ----------
    results_dict : dict
        Outer key: method label.
        Inner dict: metric -> value (expects keys from ``compute_metrics``).
    """
    COLS = [
        ("Mean 2D [m]", "mean_2d", "{:.2f}"),
        ("RMS 2D [m]", "rms_2d", "{:.2f}"),
        ("RMS 3D [m]", "rms_3d", "{:.2f}"),
        ("P50 [m]", "p50", "{:.2f}"),
        ("P67 [m]", "p67", "{:.2f}"),
        ("P95 [m]", "p95", "{:.2f}"),
        ("Max [m]", "max_2d", "{:.2f}"),
        ("N", "n_epochs", "{:d}"),
    ]

    method_width = max(len(m) for m in results_dict.keys()) + 2
    method_width = max(method_width, 20)

    # Header
    header = f"  {'Method':<{method_width}}"
    for col_name, _, _ in COLS:
        header += f"  {col_name:>12s}"
    sep = "  " + "-" * (method_width + len(COLS) * 14)

    print(sep)
    print(header)
    print(sep)

    for method, metrics in results_dict.items():
        row = f"  {method:<{method_width}}"
        for col_name, key, fmt in COLS:
            if key in metrics:
                val = metrics[key]
                try:
                    row += f"  {fmt.format(val):>12s}"
                except (TypeError, ValueError):
                    row += f"  {'N/A':>12s}"
            else:
                row += f"  {'N/A':>12s}"
        print(row)

    print(sep)


# ---------------------------------------------------------------------------
# Synthetic data generator (shared fallback across all experiments)
# ---------------------------------------------------------------------------

def generate_synthetic_urbannav(
    n_epochs: int = 300,
    n_satellites: int = 8,
    dt: float = 1.0,
    multipath_fraction: float = 0.3,
    seed: int = 42,
) -> dict:
    """Generate synthetic data mimicking UrbanNav dataset.

    Produces a vehicle trajectory through a simulated urban canyon with
    realistic multipath pseudorange errors on a fraction of satellites.

    Parameters
    ----------
    n_epochs : int
        Number of time epochs.
    n_satellites : int
        Number of satellites.
    dt : float
        Epoch interval [s].
    multipath_fraction : float
        Fraction of satellite-epoch pairs with NLOS/multipath.
    seed : int
        Random seed.

    Returns
    -------
    data : dict
        Keys: sat_ecef (N, S, 3), pseudoranges (N, S), weights (N, S),
              ground_truth (N, 3), times (N,), origin_ecef (3,),
              true_clock_bias_m (float), n_nlos_total (int).
    """
    rng = np.random.default_rng(seed)

    # Origin: near Shinjuku Station, Tokyo
    origin_lla = (np.radians(35.6896), np.radians(139.7006), 40.0)
    lat0, lon0, alt0 = origin_lla
    sin_lat = math.sin(lat0)
    cos_lat = math.cos(lat0)
    N_curv = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat ** 2)
    origin_ecef = np.array([
        (N_curv + alt0) * cos_lat * math.cos(lon0),
        (N_curv + alt0) * cos_lat * math.sin(lon0),
        (N_curv * (1.0 - _WGS84_E2) + alt0) * sin_lat,
    ])

    # ENU-to-ECEF rotation
    R_enu2ecef = np.array([
        [-math.sin(lon0), -sin_lat * math.cos(lon0), cos_lat * math.cos(lon0)],
        [math.cos(lon0), -sin_lat * math.sin(lon0), cos_lat * math.sin(lon0)],
        [0.0, cos_lat, sin_lat],
    ])

    # Vehicle trajectory: L-shaped urban drive
    speed = 5.0  # m/s
    traj_enu = np.zeros((n_epochs, 3))
    turn_epoch = n_epochs // 3 * 2
    for i in range(n_epochs):
        if i < turn_epoch:
            traj_enu[i, 0] = speed * i * dt          # east
            traj_enu[i, 1] = 0.0
        else:
            traj_enu[i, 0] = speed * turn_epoch * dt
            traj_enu[i, 1] = speed * (i - turn_epoch) * dt  # north
        traj_enu[i, 2] = 1.5

    traj_ecef = np.array([origin_ecef + R_enu2ecef @ traj_enu[i]
                          for i in range(n_epochs)])

    # Satellite positions (GPS-like, slightly changing over time)
    base_sat_enu = np.array([
        [ 15000000., -5000000., 20000000.],
        [-10000000., 12000000., 18000000.],
        [  8000000., -14000000., 19000000.],
        [ 14000000., 10000000., 16000000.],
        [-18000000., 11000000.,  7000000.],
        [  4000000., 22000000.,  8000000.],
        [ 20000000.,  3000000., 12000000.],
        [-10000000., -17000000., 12000000.],
    ], dtype=np.float64)
    # Select requested number of satellites
    base_sat_enu = base_sat_enu[:n_satellites]

    # Slow satellite motion
    sat_drift_rate = 3000.0  # m/s
    sat_ecef = np.zeros((n_epochs, n_satellites, 3))
    for i in range(n_epochs):
        t = i * dt
        for s in range(n_satellites):
            # Small circular drift
            angle = 2.0 * math.pi * t / 3600.0 * 0.5
            offset_ecef = np.array([
                sat_drift_rate * math.sin(angle + s * 0.8),
                sat_drift_rate * math.cos(angle + s * 0.8),
                sat_drift_rate * 0.1 * math.sin(angle * 2),
            ])
            sat_ecef[i, s] = origin_ecef + base_sat_enu[s] + offset_ecef

    # True receiver clock bias
    true_cb = 50000.0  # ~167 us

    # Generate clean pseudoranges
    pr_noise_los = 3.0   # m
    pr_noise_nlos = 15.0  # m
    nlos_bias_mean = 25.0  # m (positive multipath bias)

    pseudoranges = np.zeros((n_epochs, n_satellites))
    nlos_mask = np.zeros((n_epochs, n_satellites), dtype=bool)
    n_nlos_total = 0

    for i in range(n_epochs):
        for s in range(n_satellites):
            r = np.linalg.norm(sat_ecef[i, s] - traj_ecef[i])
            # NLOS based on fraction + satellite elevation heuristic
            # Low satellite index = higher elevation (less NLOS)
            nlos_prob = multipath_fraction * (1.0 + 0.5 * s / n_satellites)
            is_nlos = rng.random() < nlos_prob
            nlos_mask[i, s] = is_nlos

            if is_nlos:
                noise = rng.normal(nlos_bias_mean, pr_noise_nlos)
                n_nlos_total += 1
            else:
                noise = rng.normal(0, pr_noise_los)

            pseudoranges[i, s] = r + true_cb + noise

    # Elevation-based weights (simplified: higher weight for lower satellite index)
    weights = np.ones((n_epochs, n_satellites))
    for s in range(n_satellites):
        el_approx = math.radians(15 + 50 * (n_satellites - s) / n_satellites)
        weights[:, s] = math.sin(el_approx) ** 2

    times = np.arange(n_epochs, dtype=np.float64) * dt

    return {
        "sat_ecef": sat_ecef,
        "pseudoranges": pseudoranges,
        "weights": weights,
        "ground_truth": traj_ecef,
        "times": times,
        "satellite_counts": np.full(n_epochs, n_satellites, dtype=np.int32),
        "origin_ecef": origin_ecef,
        "true_clock_bias_m": true_cb,
        "n_nlos_total": n_nlos_total,
        "nlos_mask": nlos_mask,
        "traj_enu": traj_enu,
        "n_epochs": n_epochs,
        "n_satellites": n_satellites,
        "dt": dt,
        "source": "synthetic",
    }


# ---------------------------------------------------------------------------
# Pure-Python WLS solver (fallback)
# ---------------------------------------------------------------------------

def wls_solve_py(
    sat_ecef: np.ndarray,
    pseudoranges: np.ndarray,
    weights: np.ndarray,
    max_iter: int = 10,
    tol: float = 1e-4,
) -> tuple[np.ndarray, int]:
    """Weighted Least Squares single-epoch solver (pure Python).

    Parameters
    ----------
    sat_ecef : ndarray, shape (n_sat, 3)
    pseudoranges : ndarray, shape (n_sat,)
    weights : ndarray, shape (n_sat,)
    max_iter : int
    tol : float

    Returns
    -------
    state : ndarray, shape (4,)   [x, y, z, clock_bias]
    n_iter : int
    """
    sat_ecef = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)
    pseudoranges = np.asarray(pseudoranges, dtype=np.float64).ravel()
    weights = np.asarray(weights, dtype=np.float64).ravel()
    n_sat = len(pseudoranges)
    state = np.zeros(4, dtype=np.float64)

    if n_sat >= 4:
        centroid = np.mean(sat_ecef, axis=0)
        centroid_norm = float(np.linalg.norm(centroid))
        if centroid_norm > 1e-6:
            state[:3] = centroid * (_WGS84_A / centroid_norm)
            ranges = geometric_ranges_sagnac(state[:3], sat_ecef)
            state[3] = float(np.mean(pseudoranges - ranges))

    for iteration in range(max_iter):
        H = np.zeros((n_sat, 4))
        pred_pr = np.zeros(n_sat)
        sat_rot = rotate_satellites_sagnac(state[:3], sat_ecef)

        for j in range(n_sat):
            dx = state[:3] - sat_rot[j]
            r = np.linalg.norm(dx)
            if r < 1.0:
                r = 1.0
            pred_pr[j] = r + state[3]
            H[j, :3] = dx / r
            H[j, 3] = 1.0

        residual = pseudoranges - pred_pr
        W = np.diag(weights)

        try:
            delta = np.linalg.solve(H.T @ W @ H, H.T @ W @ residual)
        except np.linalg.LinAlgError:
            break

        state += delta
        if np.linalg.norm(delta[:3]) < tol:
            return state, iteration + 1

    return state, max_iter


# ---------------------------------------------------------------------------
# Simple CPU particle filter fallback
# ---------------------------------------------------------------------------

class SimplePFCPU:
    """Lightweight CPU particle filter for GNSS (fallback when GPU unavailable)."""

    def __init__(self, n_particles: int = 10000, sigma_pos: float = 1.0,
                 sigma_cb: float = 300.0, sigma_pr: float = 5.0,
                 resampling: str = "systematic", seed: int = 42):
        self.n = n_particles
        self.sigma_pos = sigma_pos
        self.sigma_cb = sigma_cb
        self.sigma_pr = sigma_pr
        self.resampling = resampling
        self.rng = np.random.default_rng(seed)
        self.particles = None
        self.log_w = None

    def initialize(self, pos_ecef, clock_bias=0.0, spread_pos=100.0,
                   spread_cb=1000.0):
        pos = np.asarray(pos_ecef, dtype=np.float64)
        self.particles = np.column_stack([
            self.rng.normal(pos[0], spread_pos, self.n),
            self.rng.normal(pos[1], spread_pos, self.n),
            self.rng.normal(pos[2], spread_pos, self.n),
            self.rng.normal(clock_bias, spread_cb, self.n),
        ])
        self.log_w = np.zeros(self.n)

    def predict(self, velocity=None, dt=1.0):
        if velocity is not None:
            self.particles[:, :3] += np.asarray(velocity) * dt
        self.particles[:, :3] += self.rng.normal(0, self.sigma_pos * dt,
                                                   (self.n, 3))
        self.particles[:, 3] += self.rng.normal(0, self.sigma_cb * dt, self.n)

    def update(self, sat_ecef, pseudoranges, weights=None):
        sat = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)
        pr = np.asarray(pseudoranges, dtype=np.float64)
        n_sat = len(pr)
        log_lik = np.zeros(self.n)

        for j in range(n_sat):
            dx = sat[j] - self.particles[:, :3]
            r = np.sqrt(np.sum(dx ** 2, axis=1))
            pred_pr = r + self.particles[:, 3]
            diff = pr[j] - pred_pr
            w = 1.0 if weights is None else float(np.asarray(weights)[j])
            log_lik += -0.5 * w * (diff / self.sigma_pr) ** 2

        self.log_w += log_lik
        max_lw = np.max(self.log_w)
        self.log_w -= max_lw
        w = np.exp(self.log_w)
        w /= w.sum()
        ess = 1.0 / np.sum(w ** 2)

        if ess < 0.5 * self.n:
            indices = self.rng.choice(self.n, size=self.n, p=w)
            self.particles = self.particles[indices]
            self.log_w[:] = 0.0

    def estimate(self):
        w = np.exp(self.log_w)
        w /= w.sum()
        return np.average(self.particles, axis=0, weights=w)

    def get_ess(self):
        w = np.exp(self.log_w)
        w /= w.sum()
        return float(1.0 / np.sum(w ** 2))
