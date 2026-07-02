"""Shared input validation helpers for GNSS GPU Python wrappers."""

import numpy as np


def finite_float(name, value):
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def positive_float(name, value):
    out = finite_float(name, value)
    if out <= 0.0:
        raise ValueError(f"{name} must be positive")
    return out


def nonnegative_float(name, value):
    out = finite_float(name, value)
    if out < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return out


def positive_int(name, value):
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if out != value or out < 1:
        raise ValueError(f"{name} must be a positive integer")
    return out


def as_position_ecef(position_ecef):
    arr = np.asarray(position_ecef, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError("position_ecef must have shape (3,)")
    if not np.all(np.isfinite(arr)):
        raise ValueError("position_ecef must be finite")
    return arr


def as_base_ecef(base_ecef):
    arr = np.asarray(base_ecef, dtype=np.float64).ravel()
    if arr.size != 3:
        raise ValueError("base_ecef must have shape (3,)")
    if not np.all(np.isfinite(arr)):
        raise ValueError("base_ecef must be finite")
    return arr


def finite_1d_array(name, values, *, min_size=1):
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size < min_size:
        raise ValueError(f"{name} must contain at least {min_size} value")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def nonnegative_1d_array(name, values, *, min_size=1):
    arr = finite_1d_array(name, values, min_size=min_size)
    if np.any(arr < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return arr


def as_sat_ecef_matrix(sat_ecef, n_sat):
    sat = np.asarray(sat_ecef, dtype=np.float64)
    if sat.shape != (n_sat, 3):
        raise ValueError("sat_ecef must have shape (n_sat, 3)")
    if not np.all(np.isfinite(sat)):
        raise ValueError("sat_ecef must be finite")
    return sat


def validate_gnss_observation_epoch(
    sat_ecef, pseudoranges, weights=None, min_sat=1,
):
    pr = np.asarray(pseudoranges, dtype=np.float64).ravel()
    if pr.size < min_sat:
        plural = "value" if min_sat == 1 else "values"
        raise ValueError(f"pseudoranges must contain at least {min_sat} {plural}")
    if not np.all(np.isfinite(pr)):
        raise ValueError("pseudoranges must be finite")

    n_sat = pr.size
    sat = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)
    if sat.shape != (n_sat, 3):
        raise ValueError("sat_ecef must have shape (n_sat, 3)")
    if not np.all(np.isfinite(sat)):
        raise ValueError("sat_ecef must be finite")

    if weights is None:
        weights = np.ones(n_sat, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64).ravel()
        if weights.size != n_sat:
            raise ValueError("weights length must match pseudoranges")
        if not np.all(np.isfinite(weights)):
            raise ValueError("weights must be finite")
        if np.any(weights < 0.0):
            raise ValueError("weights must be non-negative")
    return sat, pr, weights, n_sat
