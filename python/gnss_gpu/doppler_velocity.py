"""Estimate receiver velocity from GNSS Doppler observations.

Doppler shift measures the rate of change of pseudorange, which relates
to the relative velocity between receiver and satellite along the
line-of-sight direction. Given satellite positions and velocities,
the receiver velocity can be estimated via least-squares.

Reference: Groves (2013), Section 9.2.4.
"""

from __future__ import annotations

import numpy as np


# GPS L1 frequency [Hz] and speed of light [m/s]
L1_FREQ = 1575.42e6
C_LIGHT = 299792458.0


def compute_sat_velocities(
    ephemeris,
    gps_time: float,
    prn_list: list,
    dt: float = 0.5,
) -> np.ndarray | None:
    """Compute satellite velocities via numerical differentiation.

    Parameters
    ----------
    ephemeris : Ephemeris
        Ephemeris object with compute() method.
    gps_time : float
        GPS time of week [s].
    prn_list : list
        List of satellite PRN identifiers.
    dt : float
        Time step for numerical differentiation [s].

    Returns
    -------
    velocities : (K, 3) ndarray or None
        Satellite ECEF velocities [m/s].
    """
    try:
        pos_before, _, used_before = ephemeris.compute(gps_time - dt, prn_list)
        pos_after, _, used_after = ephemeris.compute(gps_time + dt, prn_list)
        if len(pos_before) != len(pos_after) or len(pos_before) == 0:
            return None
        return (pos_after - pos_before) / (2 * dt)
    except Exception:
        return None


def estimate_velocity_from_doppler(
    receiver_position: np.ndarray,
    sat_ecef: np.ndarray,
    doppler_hz: np.ndarray,
    sat_velocities: np.ndarray | None = None,
    wavelength: float = C_LIGHT / L1_FREQ,
) -> np.ndarray | None:
    """Estimate 3D receiver velocity from Doppler observations.

    Parameters
    ----------
    receiver_position : (3,) or (4,)
        Approximate receiver ECEF position [m].
    sat_ecef : (K, 3)
        Satellite ECEF positions [m].
    doppler_hz : (K,)
        Doppler observations [Hz]. Negative = approaching.
    sat_velocities : (K, 3), optional
        Satellite ECEF velocities [m/s]. If None, assumed zero
        (satellite velocity contribution not corrected).
    wavelength : float
        Carrier wavelength [m]. Default: GPS L1.

    Returns
    -------
    velocity : (3,) ndarray or None
        Estimated receiver ECEF velocity [m/s], or None if underdetermined.
    """
    sat = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)
    dop = np.asarray(doppler_hz, dtype=np.float64).ravel()
    pos = np.asarray(receiver_position, dtype=np.float64).ravel()[:3]

    # Filter valid Doppler
    valid = np.isfinite(dop) & (dop != 0.0)
    if valid.sum() < 4:
        return None

    sat = sat[valid]
    dop = dop[valid]
    n_sat = len(dop)

    # Range rate from Doppler: dr/dt = -lambda * f_doppler
    range_rate = -wavelength * dop  # [m/s]

    # Unit vectors from receiver to satellites
    dx = sat - pos[np.newaxis, :]
    ranges = np.sqrt(np.sum(dx ** 2, axis=1))
    los = dx / ranges[:, np.newaxis]  # (K, 3) line-of-sight unit vectors

    # Subtract satellite velocity contribution if available
    if sat_velocities is not None:
        sat_vel = np.asarray(sat_velocities, dtype=np.float64).reshape(-1, 3)
        sat_vel = sat_vel[valid] if len(sat_vel) > valid.sum() else sat_vel
        # Satellite range rate contribution: dot(los, sat_vel)
        sat_range_rate = np.sum(los * sat_vel, axis=1)
        range_rate -= sat_range_rate

    # Least-squares: range_rate = los @ velocity + clock_drift
    # Design matrix: [los_x, los_y, los_z, 1]
    H = np.column_stack([los, np.ones(n_sat)])
    # Solve: H @ [vx, vy, vz, clock_drift]^T = range_rate
    try:
        result, _, _, _ = np.linalg.lstsq(H, range_rate, rcond=None)
        velocity = result[:3]
        # Sanity check: reject unreasonable velocities (> 50 m/s ≈ 180 km/h)
        if np.linalg.norm(velocity) > 50.0:
            return None
        return velocity
    except np.linalg.LinAlgError:
        return None
