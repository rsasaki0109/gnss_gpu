"""GLONASS broadcast orbit propagator (RTKLIB ``geph2pos`` / ``glorbit``).

GLONASS broadcasts a 9-element initial state (ECEF position, velocity,
acceleration in PZ-90.02) rather than Keplerian elements, so the rover has to
integrate the equations of motion forward (or backward) from the reference
epoch ``toe`` to receive time.  This module owns the integrator only — record
selection, time-scale alignment, and NavMessage typing live in other modules
so the propagator can be unit-tested with synthetic state vectors.

Reference: RTKLIB ``rtklib/src/ephemeris.c`` ``geph2pos`` / ``glorbit`` / ``deq``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from gnss_gpu.gnss_time_scales import (
    GPS_WEEK_SEC,
    broadcast_rx_seconds_of_week,
)


# PZ-90 / RTKLIB ``ephemeris.c`` reference constants.
GLO_RE = 6378136.0
"""GLONASS reference equatorial radius [m]."""

GLO_MU = 3.9860044e14
"""GLONASS gravitational parameter [m^3/s^2]."""

GLO_J2 = 1.0826257e-3
"""GLONASS oblateness coefficient ``J2``."""

GLO_OMGE = 7.292115e-5
"""PZ-90 Earth-rotation rate [rad/s] (note: slightly different from WGS84)."""

GLO_EPHEMERIS_TSTEP = 60.0
"""Default RK4 integration step in seconds (RTKLIB ``TSTEP``)."""


@dataclass(frozen=True)
class GlonassBroadcastState:
    """Initial state used by the RK4 propagator.

    Position/velocity/acceleration are PZ-90 ECEF in SI units (m, m/s, m/s²).
    ``tau_n`` / ``gamma_n`` are the SV clock bias and relative-frequency bias
    (RTKLIB convention: ``clk = -tau_n + gamma_n * t``).  ``toe`` is the
    reference second-of-week aligned with the broadcast time scale.
    """

    toe: float
    px_m: float
    py_m: float
    pz_m: float
    vx_m_s: float
    vy_m_s: float
    vz_m_s: float
    ax_m_s2: float
    ay_m_s2: float
    az_m_s2: float
    tau_n: float
    gamma_n: float


def glonass_orbit_derivatives(x: np.ndarray, acc: np.ndarray) -> np.ndarray:
    """Return ``dx/dt`` for a GLONASS PZ-90 state vector.

    ``x`` is the 6-vector ``[px, py, pz, vx, vy, vz]`` in metres / metres·s⁻¹.
    ``acc`` is the 3-vector of broadcast lunar/solar acceleration in m·s⁻².
    Mirrors RTKLIB ``deq``.
    """

    xdot = np.zeros(6, dtype=np.float64)
    r2 = float(np.dot(x[:3], x[:3]))
    if r2 <= 0.0 or not math.isfinite(r2):
        return xdot
    r3 = r2 * math.sqrt(r2)
    omg2 = GLO_OMGE * GLO_OMGE
    a_term = 1.5 * GLO_J2 * GLO_MU * (GLO_RE ** 2) / r2 / r3
    b_term = 5.0 * x[2] * x[2] / r2
    c_term = -GLO_MU / r3 - a_term * (1.0 - b_term)
    xdot[0] = x[3]
    xdot[1] = x[4]
    xdot[2] = x[5]
    xdot[3] = (c_term + omg2) * x[0] + 2.0 * GLO_OMGE * x[4] + acc[0]
    xdot[4] = (c_term + omg2) * x[1] - 2.0 * GLO_OMGE * x[3] + acc[1]
    xdot[5] = (c_term - 2.0 * a_term) * x[2] + acc[2]
    return xdot


def glonass_rk4_step(step_s: float, x: np.ndarray, acc: np.ndarray) -> None:
    """Advance ``x`` by ``step_s`` seconds in-place using a classic RK4 step.

    Caller owns the array — passing the same ``x`` between calls amortises the
    allocation across the iterative propagation loop in
    :func:`propagate_glonass_state`.
    """

    k1 = glonass_orbit_derivatives(x, acc)
    w = x + k1 * (step_s / 2.0)
    k2 = glonass_orbit_derivatives(w, acc)
    w = x + k2 * (step_s / 2.0)
    k3 = glonass_orbit_derivatives(w, acc)
    w = x + k3 * step_s
    k4 = glonass_orbit_derivatives(w, acc)
    x += (k1 + 2.0 * k2 + 2.0 * k3 + k4) * (step_s / 6.0)


def _wrap_dt_to_half_week(dt: float) -> float:
    if dt > GPS_WEEK_SEC / 2.0:
        return dt - GPS_WEEK_SEC
    if dt < -GPS_WEEK_SEC / 2.0:
        return dt + GPS_WEEK_SEC
    return dt


def propagate_glonass_state(
    state: GlonassBroadcastState,
    elapsed_s: float,
    *,
    step_s: float = GLO_EPHEMERIS_TSTEP,
) -> tuple[np.ndarray, float]:
    """Integrate a GLONASS broadcast state by ``elapsed_s`` seconds.

    Returns the ECEF position [m] and clock correction [s] at the target time.
    Splits the integration into fixed-size ``step_s`` chunks with a tail step
    matching the residual so the result is independent of the sign of
    ``elapsed_s`` (RTKLIB pattern).
    """

    x = np.array(
        [
            state.px_m,
            state.py_m,
            state.pz_m,
            state.vx_m_s,
            state.vy_m_s,
            state.vz_m_s,
        ],
        dtype=np.float64,
    )
    acc = np.array(
        [state.ax_m_s2, state.ay_m_s2, state.az_m_s2],
        dtype=np.float64,
    )

    remaining = float(elapsed_s)
    base_step = -float(step_s) if remaining < 0.0 else float(step_s)
    while abs(remaining) > 1.0e-9:
        chunk = base_step if abs(remaining) >= abs(base_step) else remaining
        glonass_rk4_step(chunk, x, acc)
        remaining -= chunk

    clock = -float(state.tau_n) + float(state.gamma_n) * float(elapsed_s)
    return x[:3].copy(), clock


def glonass_position_clock(
    state: GlonassBroadcastState,
    recv_gpst_sow: float,
    *,
    step_s: float = GLO_EPHEMERIS_TSTEP,
) -> tuple[np.ndarray, float]:
    """High-level entry: align receiver GPST → UTC, propagate, return state.

    Wraps :func:`propagate_glonass_state` with the broadcast time-scale
    alignment so callers can pass GPST receive sow directly.  Mirrors RTKLIB
    ``geph2pos``.
    """

    rx_sow = broadcast_rx_seconds_of_week("R", recv_gpst_sow)
    elapsed = _wrap_dt_to_half_week(rx_sow - float(state.toe))
    return propagate_glonass_state(state, elapsed, step_s=step_s)


__all__ = [
    "GLO_RE",
    "GLO_MU",
    "GLO_J2",
    "GLO_OMGE",
    "GLO_EPHEMERIS_TSTEP",
    "GlonassBroadcastState",
    "glonass_orbit_derivatives",
    "glonass_rk4_step",
    "propagate_glonass_state",
    "glonass_position_clock",
]
