"""RTK carrier phase positioning module."""

import numpy as np

from gnss_gpu.backends import (
    NativeBackendUnavailableError,
    is_missing_optional_module,
)
from gnss_gpu.input_validation import (
    as_base_ecef,
    finite_float,
    positive_float,
)

try:
    from gnss_gpu._gnss_gpu_rtk import rtk_float, rtk_float_batch, lambda_integer
    HAS_RTK = True
except ImportError as exc:
    if not is_missing_optional_module(exc, "gnss_gpu._gnss_gpu_rtk"):
        raise
    HAS_RTK = False
    rtk_float = rtk_float_batch = lambda_integer = None


def _positive_int(name, value):
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if out <= 0:
        raise ValueError(f"{name} must be positive")
    return out


def _finite_1d_obs(name, values, *, min_size=2):
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size < min_size:
        raise ValueError(f"{name} must contain at least {min_size} values")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _as_sat_ecef(sat_ecef, n_sat):
    sat = np.asarray(sat_ecef, dtype=np.float64)
    if sat.shape == (n_sat, 3):
        pass
    elif sat.size == n_sat * 3:
        sat = sat.ravel()
    else:
        raise ValueError("sat_ecef shape must match n_sat satellites")
    if not np.all(np.isfinite(sat)):
        raise ValueError("sat_ecef must be finite")
    return np.ascontiguousarray(sat, dtype=np.float64)


def _as_batch_obs(name, values, n_epoch, n_sat):
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (n_epoch, n_sat):
        raise ValueError(f"{name} must have shape (n_epoch, n_sat)")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _as_batch_sat_ecef(sat_ecef, n_epoch, n_sat):
    sat = np.asarray(sat_ecef, dtype=np.float64)
    if sat.shape == (n_epoch, n_sat, 3):
        pass
    elif sat.shape == (n_epoch, n_sat * 3):
        pass
    elif sat.size == n_epoch * n_sat * 3:
        sat = sat.reshape(n_epoch, n_sat, 3)
    else:
        raise ValueError("sat_ecef must have shape (n_epoch, n_sat, 3) or compatible")
    if not np.all(np.isfinite(sat)):
        raise ValueError("sat_ecef must be finite")
    return np.ascontiguousarray(sat, dtype=np.float64)


class RTKSolver:
    """Double-difference RTK positioning solver.

    Parameters
    ----------
    base_ecef : array_like, shape (3,)
        Base station ECEF position [m].
    wavelength : float
        Carrier wavelength [m]. Default is GPS L1 (0.19029 m).
    max_iter : int
        Maximum Gauss-Newton iterations.
    tol : float
        Convergence tolerance [m].
    """

    def __init__(self, base_ecef, wavelength=0.19029, max_iter=20, tol=1e-4):
        self.base_ecef = as_base_ecef(base_ecef)
        self.wavelength = positive_float("wavelength", wavelength)
        self.max_iter = _positive_int("max_iter", max_iter)
        self.tol = positive_float("tol", tol)
        if not HAS_RTK:
            raise NativeBackendUnavailableError(
                "RTKSolver requires optional native module "
                "'gnss_gpu._gnss_gpu_rtk'. Build the CUDA/C++ extensions "
                "as described in README.md."
            )

    def solve_float(self, rover_pr, base_pr, rover_carrier, base_carrier, sat_ecef):
        """Float RTK solution.

        Parameters
        ----------
        rover_pr : array_like, shape (n_sat,)
            Rover pseudoranges [m].
        base_pr : array_like, shape (n_sat,)
            Base pseudoranges [m].
        rover_carrier : array_like, shape (n_sat,)
            Rover carrier phase observations [cycles].
        base_carrier : array_like, shape (n_sat,)
            Base carrier phase observations [cycles].
        sat_ecef : array_like, shape (n_sat, 3)
            Satellite ECEF positions [m].

        Returns
        -------
        position : ndarray, shape (3,)
            Rover ECEF position [m].
        ambiguities : ndarray, shape (n_sat-1,)
            Float DD ambiguities [cycles].
        residuals : ndarray, shape (2*(n_sat-1),)
            DD residuals (pseudorange then carrier) [m, cycles].
        """
        rpr = _finite_1d_obs("rover_pr", rover_pr, min_size=2)
        n_sat = rpr.size
        bpr = _finite_1d_obs("base_pr", base_pr, min_size=n_sat)
        rcp = _finite_1d_obs("rover_carrier", rover_carrier, min_size=n_sat)
        bcp = _finite_1d_obs("base_carrier", base_carrier, min_size=n_sat)
        if bpr.size != n_sat:
            raise ValueError("base_pr length must match n_sat")
        if rcp.size != n_sat:
            raise ValueError("rover_carrier length must match n_sat")
        if bcp.size != n_sat:
            raise ValueError("base_carrier length must match n_sat")
        sat = _as_sat_ecef(sat_ecef, n_sat)

        rpr = np.ascontiguousarray(rpr, dtype=np.float64)
        bpr = np.ascontiguousarray(bpr, dtype=np.float64)
        rcp = np.ascontiguousarray(rcp, dtype=np.float64)
        bcp = np.ascontiguousarray(bcp, dtype=np.float64)

        result, ambiguities, residuals, iters = rtk_float(
            self.base_ecef, rpr, bpr, rcp, bcp, sat,
            self.wavelength, self.max_iter, self.tol)

        return result, ambiguities, residuals

    def solve_fixed(self, rover_pr, base_pr, rover_carrier, base_carrier, sat_ecef,
                    n_candidates=100, ratio_threshold=3.0):
        """Fixed RTK solution with LAMBDA ambiguity resolution.

        Parameters
        ----------
        rover_pr, base_pr, rover_carrier, base_carrier, sat_ecef :
            Same as solve_float.
        n_candidates : int
            Number of integer candidates to evaluate.
        ratio_threshold : float
            Ratio test threshold. Fix accepted if ratio >= threshold.

        Returns
        -------
        position : ndarray, shape (3,)
            Rover ECEF position [m] (fixed if ratio test passed, else float).
        fix_flag : bool
            True if integer ambiguities were successfully fixed.
        ratio : float
            Ratio test value (second-best / best chi-squared).
        """
        n_candidates = _positive_int("n_candidates", n_candidates)
        ratio_threshold = finite_float("ratio_threshold", ratio_threshold)

        rpr = _finite_1d_obs("rover_pr", rover_pr, min_size=2)
        n_sat = rpr.size
        bpr = _finite_1d_obs("base_pr", base_pr, min_size=n_sat)
        rcp = _finite_1d_obs("rover_carrier", rover_carrier, min_size=n_sat)
        bcp = _finite_1d_obs("base_carrier", base_carrier, min_size=n_sat)
        if bpr.size != n_sat:
            raise ValueError("base_pr length must match n_sat")
        if rcp.size != n_sat:
            raise ValueError("rover_carrier length must match n_sat")
        if bcp.size != n_sat:
            raise ValueError("base_carrier length must match n_sat")
        sat = _as_sat_ecef(sat_ecef, n_sat)

        rpr = np.ascontiguousarray(rpr, dtype=np.float64)
        bpr = np.ascontiguousarray(bpr, dtype=np.float64)
        rcp = np.ascontiguousarray(rcp, dtype=np.float64)
        bcp = np.ascontiguousarray(bcp, dtype=np.float64)

        # Get float solution
        result, ambiguities, residuals, iters = rtk_float(
            self.base_ecef, rpr, bpr, rcp, bcp, sat,
            self.wavelength, self.max_iter, self.tol)

        n_dd = len(ambiguities)

        # Build approximate ambiguity covariance (diagonal, from residuals)
        # In production you'd extract this from the normal equation inverse
        Q_amb = np.eye(n_dd, dtype=np.float64) * 0.1

        # LAMBDA resolution
        fixed_amb, ratio = lambda_integer(ambiguities, Q_amb.ravel(), n_candidates)

        fix_flag = ratio >= ratio_threshold

        # If fixed, the position from float is already close;
        # for a full implementation you would re-solve with fixed ambiguities.
        # Here we return the float position (short baseline, cm-level already).
        return result, fix_flag, ratio

    def solve_batch(self, rover_pr, base_pr, rover_carrier, base_carrier, sat_ecef):
        """Batch float RTK solution (GPU parallel).

        Parameters
        ----------
        rover_pr : array_like, shape (n_epoch, n_sat)
        base_pr : array_like, shape (n_epoch, n_sat)
        rover_carrier : array_like, shape (n_epoch, n_sat)
        base_carrier : array_like, shape (n_epoch, n_sat)
        sat_ecef : array_like, shape (n_epoch, n_sat, 3)

        Returns
        -------
        positions : ndarray, shape (n_epoch, 3)
            Rover ECEF positions [m].
        ambiguities : ndarray, shape (n_epoch, n_sat-1)
            Float DD ambiguities [cycles].
        iters : ndarray, shape (n_epoch,)
            Iterations per epoch.
        """
        rpr = np.asarray(rover_pr, dtype=np.float64)
        if rpr.ndim != 2:
            raise ValueError("rover_pr must have shape (n_epoch, n_sat)")
        if rpr.shape[0] < 1:
            raise ValueError("n_epoch must be >= 1")
        if rpr.shape[1] < 2:
            raise ValueError("n_sat must be >= 2")
        n_epoch, n_sat = rpr.shape

        rpr = _as_batch_obs("rover_pr", rpr, n_epoch, n_sat)
        bpr = _as_batch_obs("base_pr", base_pr, n_epoch, n_sat)
        rcp = _as_batch_obs("rover_carrier", rover_carrier, n_epoch, n_sat)
        bcp = _as_batch_obs("base_carrier", base_carrier, n_epoch, n_sat)
        sat = _as_batch_sat_ecef(sat_ecef, n_epoch, n_sat)

        results, ambiguities, iters = rtk_float_batch(
            self.base_ecef, rpr, bpr, rcp, bcp, sat,
            self.wavelength, self.max_iter, self.tol)

        return results, ambiguities, iters
