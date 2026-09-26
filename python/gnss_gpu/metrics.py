"""Generic coordinate conversions and positioning accuracy metrics."""

from __future__ import annotations

import math

import numpy as np


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
