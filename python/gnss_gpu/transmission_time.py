"""Transmission-time / Sagnac satellite-position correction.

A pseudorange is emitted ~0.07 s before it is received. Computing the satellite
ECEF position at the *reception* epoch therefore places it where the satellite
*is now*, not where it *was* when the signal left it; the geometric range then
carries a per-satellite error of (range-rate x travel-time) -- tens of metres --
that swamps the multipath/NLOS signal in pseudorange residuals.

The fix recomputes each satellite at ``tow - range/c`` (the transmission epoch)
and rotates the result by the Earth rotation during the signal's flight (the
Sagnac term), expressing it in the ECEF frame at reception time before a range
is formed.

This module is the canonical home for the correction so that both the dataset
loader (:mod:`gnss_gpu.io.urbannav`) and the residual tooling
(:mod:`gnss_gpu.validation.real_residuals`) share one implementation.
"""

from __future__ import annotations

import math

import numpy as np

C_LIGHT = 299792458.0
EARTH_ROTATION_RATE = 7.2921151467e-5  # rad/s (WGS-84 / GPS)


def sagnac_rotate(position, travel_time_s, omega_e: float = EARTH_ROTATION_RATE):
    """Rotate an ECEF position by the Earth rotation during signal travel.

    A satellite position computed in the ECEF frame at *transmission* time must
    be expressed in the ECEF frame at *reception* time before forming a range to
    the receiver; the frame rotates by ``omega_e * travel_time`` about +Z. Returns
    the rotated position (same shape as the [3] input).
    """
    p = np.asarray(position, dtype=float).reshape(3)
    theta = float(omega_e) * float(travel_time_s)
    c, s = math.cos(theta), math.sin(theta)
    return np.array([c * p[0] + s * p[1], -s * p[0] + c * p[1], p[2]], dtype=float)


def transmission_time_sat_positions(
    eph,
    tow: float,
    rx_ecef,
    sat_ids,
    sat_ecef_reception,
    *,
    obs_codes=None,
    omega_e: float = EARTH_ROTATION_RATE,
):
    """Satellite ECEF positions at signal *transmission* time, Sagnac-corrected.

    ``eph`` is a :class:`gnss_gpu.ephemeris.Ephemeris`; ``sat_ecef_reception`` are
    reception-time positions (used only to seed the travel time). Each satellite
    is recomputed at ``tow - range/c`` (one iteration from the reception-time
    range, accurate to <1 m) and the Sagnac rotation is applied. Returns an
    [n_sat, 3] array; rows for satellites the ephemeris cannot evaluate at the
    transmission epoch are filled with NaN.
    """
    rx = np.asarray(rx_ecef, dtype=float).reshape(3)
    sat0 = np.asarray(sat_ecef_reception, dtype=float).reshape(-1, 3)
    out = np.full((len(sat_ids), 3), np.nan, dtype=float)
    for i, sid in enumerate(sat_ids):
        tau = float(np.linalg.norm(sat0[i] - rx)) / C_LIGHT
        codes = None if obs_codes is None else [obs_codes[i]]
        se, _clk, used = eph.compute(float(tow) - tau, [sid], obs_codes=codes)
        se = np.asarray(se, dtype=float).reshape(-1, 3)
        if se.shape[0] == 0:
            continue
        out[i] = sagnac_rotate(se[0], tau, omega_e)
    return out


__all__ = [
    "C_LIGHT",
    "EARTH_ROTATION_RATE",
    "sagnac_rotate",
    "transmission_time_sat_positions",
]
