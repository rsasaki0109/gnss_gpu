from __future__ import annotations

import math

import numpy as np
import pytest

from gnss_gpu.orbit_glonass import (
    GLO_EPHEMERIS_TSTEP,
    GLO_MU,
    GLO_OMGE,
    GlonassBroadcastState,
    glonass_orbit_derivatives,
    glonass_position_clock,
    glonass_rk4_step,
    propagate_glonass_state,
)


# Canonical GLONASS MEO radius (PZ-90).  Real ephemerides hover near here.
_GLO_RADIUS_M = 2.55e7


def _circular_state(toe: float = 0.0) -> GlonassBroadcastState:
    """Build a synthetic on-orbit GLONASS state for sanity tests."""

    return GlonassBroadcastState(
        toe=toe,
        px_m=_GLO_RADIUS_M,
        py_m=0.0,
        pz_m=0.0,
        # Roughly circular speed √(μ/r) — orbit will not stay perfectly
        # circular under J2 + Coriolis, but the magnitude is correct.
        vx_m_s=0.0,
        vy_m_s=math.sqrt(GLO_MU / _GLO_RADIUS_M),
        vz_m_s=0.0,
        ax_m_s2=0.0,
        ay_m_s2=0.0,
        az_m_s2=0.0,
        tau_n=1.0e-6,
        gamma_n=0.0,
    )


# --- glonass_orbit_derivatives -------------------------------------------


def test_orbit_derivatives_zero_state_returns_zero_velocity_terms():
    x = np.zeros(6, dtype=np.float64)
    acc = np.zeros(3, dtype=np.float64)
    out = glonass_orbit_derivatives(x, acc)
    np.testing.assert_array_equal(out, np.zeros(6))


def test_orbit_derivatives_copies_velocity_into_position_slot():
    x = np.array(
        [_GLO_RADIUS_M, 0.0, 0.0, 1.0, 2.0, 3.0],
        dtype=np.float64,
    )
    acc = np.zeros(3, dtype=np.float64)
    out = glonass_orbit_derivatives(x, acc)
    # First three components of d/dt x should equal the velocity slot.
    np.testing.assert_array_equal(out[:3], np.array([1.0, 2.0, 3.0]))


def test_orbit_derivatives_acceleration_combines_gravity_and_centrifugal():
    x = np.array([_GLO_RADIUS_M, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    acc = np.zeros(3, dtype=np.float64)
    out = glonass_orbit_derivatives(x, acc)
    # Central pull -μ/r² ≈ -0.613 m/s² minus centrifugal +ω²r ≈ +0.136 m/s²
    # (Earth-rotating frame); J2 contribution is negligible at the equator.
    central = -GLO_MU / (_GLO_RADIUS_M ** 2)
    centrifugal = (GLO_OMGE ** 2) * _GLO_RADIUS_M
    assert out[3] == pytest.approx(central + centrifugal, rel=0.01)
    # The aggregate must remain negative (gravity dominates centrifugal here).
    assert out[3] < 0.0


# --- glonass_rk4_step ----------------------------------------------------


def test_rk4_step_mutates_in_place():
    x = np.array(
        [_GLO_RADIUS_M, 0.0, 0.0, 0.0, math.sqrt(GLO_MU / _GLO_RADIUS_M), 0.0],
        dtype=np.float64,
    )
    acc = np.zeros(3, dtype=np.float64)
    x_before = x.copy()
    glonass_rk4_step(60.0, x, acc)
    # x must have been updated (RK4 step is non-trivial for non-zero state).
    assert not np.array_equal(x, x_before)


def test_rk4_step_with_zero_dt_is_identity():
    x = np.array(
        [_GLO_RADIUS_M, 0.0, 0.0, 0.0, math.sqrt(GLO_MU / _GLO_RADIUS_M), 0.0],
        dtype=np.float64,
    )
    acc = np.zeros(3, dtype=np.float64)
    x_before = x.copy()
    glonass_rk4_step(0.0, x, acc)
    np.testing.assert_array_equal(x, x_before)


# --- propagate_glonass_state ---------------------------------------------


def test_propagate_zero_elapsed_returns_initial_position_and_no_clock_drift():
    state = _circular_state()
    pos, clk = propagate_glonass_state(state, elapsed_s=0.0)
    np.testing.assert_allclose(pos, [_GLO_RADIUS_M, 0.0, 0.0])
    # clk = -tau_n + gamma_n * elapsed = -tau_n
    assert clk == pytest.approx(-state.tau_n)


def test_propagate_clock_correction_includes_gamma_drift():
    state = GlonassBroadcastState(
        toe=0.0,
        px_m=_GLO_RADIUS_M,
        py_m=0.0,
        pz_m=0.0,
        vx_m_s=0.0,
        vy_m_s=math.sqrt(GLO_MU / _GLO_RADIUS_M),
        vz_m_s=0.0,
        ax_m_s2=0.0,
        ay_m_s2=0.0,
        az_m_s2=0.0,
        tau_n=1.0e-6,
        gamma_n=1.0e-9,
    )
    _, clk = propagate_glonass_state(state, elapsed_s=60.0)
    assert clk == pytest.approx(-state.tau_n + state.gamma_n * 60.0)


def test_propagate_radius_is_quasi_conserved_under_central_orbit():
    state = _circular_state()
    pos, _ = propagate_glonass_state(state, elapsed_s=300.0)
    # Radius should stay within ~1 % over 5 minutes despite J2 + Coriolis.
    r = float(np.linalg.norm(pos))
    assert r == pytest.approx(_GLO_RADIUS_M, rel=0.01)


def test_propagate_negative_elapsed_runs_integration_backwards():
    state = _circular_state()
    pos_forward, _ = propagate_glonass_state(state, elapsed_s=120.0)
    pos_backward, _ = propagate_glonass_state(state, elapsed_s=-120.0)
    # Forward and backward 2-minute propagations land on different in-plane points.
    assert not np.allclose(pos_forward, pos_backward, atol=1.0)


def test_propagate_step_size_overrideable():
    state = _circular_state()
    pos_default, _ = propagate_glonass_state(state, elapsed_s=120.0)
    pos_finer, _ = propagate_glonass_state(state, elapsed_s=120.0, step_s=30.0)
    # 60s vs 30s RK4: 4th-order error makes the two agree within 1 m at 2 minutes.
    np.testing.assert_allclose(pos_default, pos_finer, atol=1.0)


def test_propagate_default_step_equals_documented_constant():
    assert GLO_EPHEMERIS_TSTEP == 60.0


# --- glonass_position_clock ----------------------------------------------


def test_glonass_position_clock_applies_gpst_to_utc_alignment():
    # If we ask for the satellite position at recv_gpst = toe + 18s
    # (matching the GPST→UTC offset) the elapsed broadcast time is zero,
    # i.e. we should recover the initial state.
    state = _circular_state(toe=12345.0)
    pos, clk = glonass_position_clock(state, recv_gpst_sow=state.toe + 18.0)
    np.testing.assert_allclose(pos, [_GLO_RADIUS_M, 0.0, 0.0])
    assert clk == pytest.approx(-state.tau_n)


def test_glonass_position_clock_propagates_after_alignment():
    state = _circular_state(toe=12345.0)
    pos_now, _ = glonass_position_clock(state, recv_gpst_sow=state.toe + 18.0)
    pos_later, _ = glonass_position_clock(state, recv_gpst_sow=state.toe + 18.0 + 300.0)
    # After 5 minutes, the satellite should have moved measurably along ŷ.
    assert pos_later[1] - pos_now[1] > 1.0e5  # > 100 km along orbit
