"""Robust Single Point Positioning (SPP) via Iteratively Reweighted Least Squares.

Provides a Python-level robust WLS solver that can replace the standard
gnssplusplus SPP when NLOS outliers are present. Uses Huber or Cauchy weight
functions to downweight outlier satellites.
"""

from __future__ import annotations

import numpy as np

# Speed of light [m/s] - not needed for position-domain WLS but kept for reference
_C = 299_792_458.0


def robust_spp(
    sat_ecef: np.ndarray,
    pseudoranges: np.ndarray,
    weights: np.ndarray | None = None,
    init_pos: np.ndarray | None = None,
    max_iter: int = 10,
    threshold: float = 15.0,
    weight_func: str = "cauchy",
    convergence_m: float = 1e-4,
    min_satellites: int = 5,
) -> np.ndarray | None:
    """Iteratively reweighted least squares SPP with outlier downweighting.

    Parameters
    ----------
    sat_ecef : ndarray, shape (n_sat, 3)
        Satellite ECEF positions [m].
    pseudoranges : ndarray, shape (n_sat,)
        Corrected pseudoranges [m].
    weights : ndarray, shape (n_sat,), optional
        Per-satellite weights from elevation model etc. Defaults to ones.
    init_pos : ndarray, shape (3,), optional
        Initial position estimate [m] (e.g. from standard SPP).
        If None, uses geometric center of satellites as starting point.
    max_iter : int
        Maximum number of IRLS iterations.
    threshold : float
        Residual threshold [m] for robust weight function.
    weight_func : str
        "cauchy" or "huber".
    convergence_m : float
        Convergence criterion on position update norm [m].
    min_satellites : int
        Minimum satellites required after outlier exclusion (must be >= 4).

    Returns
    -------
    position : ndarray, shape (3,) or None
        ECEF position [m], or None if underdetermined / failed to converge.
    """
    _validate_solver_options(max_iter, threshold, weight_func, convergence_m, min_satellites)

    sat = _as_satellite_matrix(sat_ecef)
    pr = _as_vector(pseudoranges, "pseudoranges")
    n_sat = len(pr)

    if len(sat) != n_sat:
        raise ValueError(
            f"sat_ecef has {len(sat)} satellites but pseudoranges has {n_sat} values"
        )

    if n_sat < 4:
        return None

    if weights is None:
        w_elev = np.ones(n_sat, dtype=np.float64)
    else:
        w_elev = _as_vector(weights, "weights").copy()
        if len(w_elev) != n_sat:
            raise ValueError(f"weights has {len(w_elev)} values but pseudoranges has {n_sat}")
        if np.any(w_elev < 0.0):
            raise ValueError("weights must be non-negative")

    # State: [x, y, z, clock_bias]
    if init_pos is not None:
        pos = _as_initial_position(init_pos)
    else:
        pos = sat.mean(axis=0).copy()

    # Estimate initial clock bias
    ranges = np.linalg.norm(sat - pos, axis=1)
    cb = float(np.median(pr - ranges))

    state = np.array([pos[0], pos[1], pos[2], cb], dtype=np.float64)

    for iteration in range(max_iter):
        dx_vec = sat - state[:3]
        ranges = np.linalg.norm(dx_vec, axis=1)

        # Avoid division by zero
        ranges = np.maximum(ranges, 1.0)

        # Line-of-sight unit vectors
        los = dx_vec / ranges[:, np.newaxis]

        # Design matrix H: [-los_x, -los_y, -los_z, 1]
        H = np.column_stack([-los, np.ones(n_sat)])

        # Observation residuals: observed - predicted
        predicted = ranges + state[3]
        y = pr - predicted

        # Compute robust weights based on residuals (after first iteration)
        if iteration == 0:
            w_robust = np.ones(n_sat, dtype=np.float64)
        else:
            w_robust = _compute_robust_weights(y, threshold, weight_func)

        # Combined weight: elevation * robust
        w_total = w_elev * w_robust

        # Check we have enough well-weighted satellites
        effective_sats = np.sum(w_total > 0.1)
        if effective_sats < min_satellites:
            return None

        # Weighted least squares: (H^T W H)^{-1} H^T W y
        W = np.diag(w_total)
        HTWH = H.T @ W @ H
        HTWy = H.T @ W @ y

        try:
            delta = np.linalg.solve(HTWH, HTWy)
        except np.linalg.LinAlgError:
            return None

        state += delta

        # Check convergence (position update only)
        if np.linalg.norm(delta[:3]) < convergence_m:
            break

    return state[:3].copy()


def _as_satellite_matrix(value: np.ndarray) -> np.ndarray:
    """Return satellite coordinates as a finite ``(n_sat, 3)`` float array."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 1:
        if arr.size % 3 != 0:
            raise ValueError("sat_ecef flat input length must be divisible by 3")
        arr = arr.reshape(-1, 3)
    elif arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("sat_ecef must have shape (n_sat, 3) or be a flat xyz array")
    if not np.all(np.isfinite(arr)):
        raise ValueError("sat_ecef must contain only finite values")
    return arr


def _as_vector(value: np.ndarray, name: str) -> np.ndarray:
    """Return a finite 1-D float vector."""
    arr = np.asarray(value, dtype=np.float64).ravel()
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _as_initial_position(value: np.ndarray) -> np.ndarray:
    arr = _as_vector(value, "init_pos")
    if len(arr) < 3:
        raise ValueError("init_pos must contain at least 3 values")
    return arr[:3].copy()


def _validate_solver_options(
    max_iter: int,
    threshold: float,
    weight_func: str,
    convergence_m: float,
    min_satellites: int,
) -> None:
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("threshold must be a finite positive value")
    if not np.isfinite(convergence_m) or convergence_m <= 0.0:
        raise ValueError("convergence_m must be a finite positive value")
    if min_satellites < 4:
        raise ValueError("min_satellites must be >= 4")
    if weight_func not in {"cauchy", "huber"}:
        raise ValueError(f"Unknown weight function: {weight_func}")


def _compute_robust_weights(
    residuals: np.ndarray,
    threshold: float,
    weight_func: str,
) -> np.ndarray:
    """Compute robust weights from residuals.

    Parameters
    ----------
    residuals : ndarray
        Observation residuals [m].
    threshold : float
        Scale parameter [m].
    weight_func : str
        "cauchy" or "huber".

    Returns
    -------
    weights : ndarray
        Robust weights in (0, 1].
    """
    r = np.abs(residuals)
    if weight_func == "cauchy":
        return 1.0 / (1.0 + (r / threshold) ** 2)
    elif weight_func == "huber":
        w = np.ones_like(r)
        mask = r > threshold
        w[mask] = threshold / r[mask]
        return w
    else:
        raise ValueError(f"Unknown weight function: {weight_func}")
