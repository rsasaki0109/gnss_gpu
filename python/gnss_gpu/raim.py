"""RAIM (Receiver Autonomous Integrity Monitoring) and FDE (Fault Detection and Exclusion).

Provides integrity monitoring for GNSS positioning by detecting and excluding
faulty satellite measurements.
"""

import numpy as np

try:
    from gnss_gpu._gnss_gpu_raim import RAIMResult, raim_check as _raim_check, raim_fde as _raim_fde
    HAS_RAIM = True
except ImportError:
    HAS_RAIM = False


def _no_redundancy_result():
    result = RAIMResult()
    result.integrity_ok = True
    result.hpl = 1e9
    result.vpl = 1e9
    result.test_statistic = 0.0
    result.threshold = 0.0
    result.excluded_sat = -1
    return result


def _validate_raim_inputs(name, sat_ecef, pseudoranges, weights, position, p_fa):
    pseudoranges = np.asarray(pseudoranges, dtype=np.float64)
    if pseudoranges.ndim != 1:
        raise RuntimeError(f"{name}: pseudoranges must have shape (n_sat,)")
    n_sat = pseudoranges.size
    if n_sat < 4:
        raise RuntimeError(f"{name} requires at least 4 satellites")

    sat_ecef = np.asarray(sat_ecef, dtype=np.float64)
    if sat_ecef.ndim == 1 and sat_ecef.size == n_sat * 3:
        sat_ecef = sat_ecef.reshape(n_sat, 3)
    elif not (sat_ecef.ndim == 2 and sat_ecef.shape == (n_sat, 3)):
        raise RuntimeError(f"{name}: sat_ecef must have shape (n_sat, 3) or flat length n_sat*3")

    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or weights.size != n_sat:
        raise RuntimeError(f"{name}: weights must have shape (n_sat,)")

    position = np.asarray(position, dtype=np.float64)
    if position.ndim != 1 or position.size != 4:
        raise RuntimeError(f"{name}: position must have shape (4,)")

    if not np.isfinite(p_fa) or p_fa <= 0.0 or p_fa >= 1.0:
        raise RuntimeError(f"{name}: p_fa must be in (0, 1)")
    if not np.isfinite(sat_ecef).all():
        raise RuntimeError(f"{name}: satellite positions must be finite")
    if not np.isfinite(pseudoranges).all():
        raise RuntimeError(f"{name}: pseudoranges must be finite")
    if not (np.isfinite(weights).all() and np.all(weights >= 0.0)):
        raise RuntimeError(f"{name}: weights must be finite and nonnegative")
    if not np.isfinite(position).all():
        raise RuntimeError(f"{name}: position must be finite")

    return (
        np.ascontiguousarray(sat_ecef, dtype=np.float64),
        np.ascontiguousarray(pseudoranges, dtype=np.float64),
        np.ascontiguousarray(weights, dtype=np.float64),
        np.ascontiguousarray(position, dtype=np.float64),
    )


def raim_check(sat_ecef, pseudoranges, weights, position, p_fa=1e-5):
    """Run RAIM chi-squared consistency check.

    Args:
        sat_ecef: (n_sat, 3) or flat length n_sat*3 satellite ECEF positions [m].
        pseudoranges: (n_sat,) observed pseudoranges [m].
        weights: (n_sat,) observation weights (1/sigma^2).
        position: (4,) WLS solution [x, y, z, clock_bias] in ECEF [m].
        p_fa: Probability of false alarm (default 1e-5).

    Returns:
        RAIMResult with integrity_ok, hpl, vpl, test_statistic, threshold, excluded_sat.
    """
    sat_ecef, pseudoranges, weights, position = _validate_raim_inputs(
        "raim_check", sat_ecef, pseudoranges, weights, position, p_fa
    )

    if not HAS_RAIM:
        raise RuntimeError("RAIM native module not available. Build with CUDA support.")

    n_sat = pseudoranges.size
    if n_sat == 4:
        return _no_redundancy_result()

    return _raim_check(sat_ecef, pseudoranges, weights, position, p_fa)


def raim_fde(sat_ecef, pseudoranges, weights, position, p_fa=1e-5):
    """Run RAIM with Fault Detection and Exclusion.

    If the consistency check fails, tries excluding each satellite in turn,
    re-solves WLS, and selects the exclusion that yields the lowest SSE.

    Args:
        sat_ecef: (n_sat, 3) or flat length n_sat*3 satellite ECEF positions [m].
        pseudoranges: (n_sat,) observed pseudoranges [m].
        weights: (n_sat,) observation weights (1/sigma^2).
        position: (4,) WLS solution [x, y, z, clock_bias] in ECEF [m].
        p_fa: Probability of false alarm (default 1e-5).

    Returns:
        Tuple of (RAIMResult, position_array).
        If a satellite was excluded, position_array contains the corrected solution.
    """
    sat_ecef, pseudoranges, weights, position = _validate_raim_inputs(
        "raim_fde", sat_ecef, pseudoranges, weights, position, p_fa
    )

    if not HAS_RAIM:
        raise RuntimeError("RAIM native module not available. Build with CUDA support.")

    n_sat = pseudoranges.size
    if n_sat > 64:
        raise RuntimeError("raim_fde supports at most 64 satellites")
    if n_sat == 4:
        return _no_redundancy_result(), position.copy()

    return _raim_fde(sat_ecef, pseudoranges, weights, position, p_fa)
