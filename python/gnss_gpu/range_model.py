"""Shared GNSS geometric range models."""

from __future__ import annotations

import numpy as np

C_LIGHT = 299792458.0
OMEGA_E = 7.2921151467e-5


def rotate_satellites_sagnac(
    receiver_ecef: np.ndarray,
    sat_ecef: np.ndarray,
) -> np.ndarray:
    """Rotate satellite ECEF positions for Earth rotation during signal transit."""

    rx = np.asarray(receiver_ecef, dtype=np.float64).reshape(3)
    sat = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)

    delta0 = rx.reshape(1, 3) - sat
    range_approx = np.linalg.norm(delta0, axis=1)
    theta = OMEGA_E * (range_approx / C_LIGHT)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    sat_rot = sat.copy()
    sat_rot[:, 0] = sat[:, 0] * cos_t + sat[:, 1] * sin_t
    sat_rot[:, 1] = -sat[:, 0] * sin_t + sat[:, 1] * cos_t
    return sat_rot


def geometric_ranges_sagnac(
    receiver_ecef: np.ndarray,
    sat_ecef: np.ndarray,
) -> np.ndarray:
    """Return receiver-to-satellite ranges using the native WLS Sagnac model."""

    rx = np.asarray(receiver_ecef, dtype=np.float64).reshape(3)
    sat_rot = rotate_satellites_sagnac(rx, sat_ecef)
    return np.linalg.norm(rx.reshape(1, 3) - sat_rot, axis=1)
