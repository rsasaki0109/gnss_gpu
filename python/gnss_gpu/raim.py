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


def raim_check(sat_ecef, pseudoranges, weights, position, p_fa=1e-5):
    """Run RAIM chi-squared consistency check.

    Args:
        sat_ecef: (n_sat, 3) satellite ECEF positions [m].
        pseudoranges: (n_sat,) observed pseudoranges [m].
        weights: (n_sat,) observation weights (1/sigma^2).
        position: (4,) WLS solution [x, y, z, clock_bias] in ECEF [m].
        p_fa: Probability of false alarm (default 1e-5).

    Returns:
        RAIMResult with integrity_ok, hpl, vpl, test_statistic, threshold, excluded_sat.
    """
    if not HAS_RAIM:
        raise RuntimeError("RAIM native module not available. Build with CUDA support.")

    sat_ecef = np.ascontiguousarray(sat_ecef, dtype=np.float64).flatten()
    pseudoranges = np.ascontiguousarray(pseudoranges, dtype=np.float64)
    weights = np.ascontiguousarray(weights, dtype=np.float64)
    position = np.ascontiguousarray(position, dtype=np.float64)

    n_sat = pseudoranges.size
    if n_sat < 4:
        raise RuntimeError("raim_check requires at least 4 satellites")
    if n_sat == 4:
        return _no_redundancy_result()

    return _raim_check(sat_ecef, pseudoranges, weights, position, p_fa)


def raim_fde(sat_ecef, pseudoranges, weights, position, p_fa=1e-5):
    """Run RAIM with Fault Detection and Exclusion.

    If the consistency check fails, tries excluding each satellite in turn,
    re-solves WLS, and selects the exclusion that yields the lowest SSE.

    Args:
        sat_ecef: (n_sat, 3) satellite ECEF positions [m].
        pseudoranges: (n_sat,) observed pseudoranges [m].
        weights: (n_sat,) observation weights (1/sigma^2).
        position: (4,) WLS solution [x, y, z, clock_bias] in ECEF [m].
        p_fa: Probability of false alarm (default 1e-5).

    Returns:
        Tuple of (RAIMResult, position_array).
        If a satellite was excluded, position_array contains the corrected solution.
    """
    if not HAS_RAIM:
        raise RuntimeError("RAIM native module not available. Build with CUDA support.")

    sat_ecef = np.ascontiguousarray(sat_ecef, dtype=np.float64).flatten()
    pseudoranges = np.ascontiguousarray(pseudoranges, dtype=np.float64)
    weights = np.ascontiguousarray(weights, dtype=np.float64)
    position = np.ascontiguousarray(position, dtype=np.float64)

    n_sat = pseudoranges.size
    if n_sat < 4:
        raise RuntimeError("raim_fde requires at least 4 satellites")
    if n_sat == 4:
        return _no_redundancy_result(), position.copy()

    return _raim_fde(sat_ecef, pseudoranges, weights, position, p_fa)
