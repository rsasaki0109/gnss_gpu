"""Tests for TDCP velocity estimation."""

from __future__ import annotations

import numpy as np
import pytest

from gnss_gpu.doppler_velocity import C_LIGHT, L1_FREQ
from gnss_gpu.tdcp_velocity import (
    L1_WAVELENGTH,
    _one_row_per_satellite,
    estimate_velocity_from_tdcp,
    estimate_velocity_from_tdcp_with_metrics,
)

LAM = C_LIGHT / L1_FREQ


class _Meas:
    __slots__ = (
        "system_id",
        "prn",
        "satellite_ecef",
        "carrier_phase",
        "satellite_velocity",
        "clock_drift",
        "weight",
        "snr",
        "elevation",
    )

    def __init__(
        self,
        system_id: int,
        prn: int,
        satellite_ecef: np.ndarray,
        carrier_phase: float,
        satellite_velocity: np.ndarray | None = None,
        clock_drift: float = 0.0,
        weight: float = 1.0,
        snr: float = 45.0,
        elevation: float = float("nan"),
    ):
        self.system_id = system_id
        self.prn = prn
        self.satellite_ecef = np.asarray(satellite_ecef, dtype=np.float64)
        self.carrier_phase = float(carrier_phase)
        self.satellite_velocity = (
            np.zeros(3, dtype=np.float64)
            if satellite_velocity is None
            else np.asarray(satellite_velocity, dtype=np.float64)
        )
        self.clock_drift = float(clock_drift)
        self.weight = float(weight)
        self.snr = float(snr)
        self.elevation = float(elevation)


def _los(rx: np.ndarray, sat: np.ndarray) -> np.ndarray:
    d = sat - rx
    return d / np.linalg.norm(d)


def test_tdcp_recovers_constant_velocity_zero_sat_motion():
    rng = np.random.default_rng(0)
    rx = np.array([1.0e6, 2.0e6, 3.0e6])
    dt = 1.0
    delta_rx = np.array([5.1, -2.3, 0.8])
    delta_cb = 1.5

    sats = [
        rx + np.array([20e6, 0.0, 0.0]),
        rx + np.array([0.0, 21e6, 0.0]),
        rx + np.array([0.0, 0.0, 22e6]),
        rx + np.array([15e6, 15e6, 10e6]),
    ]

    prev: list[_Meas] = []
    cur: list[_Meas] = []
    base_cycles = rng.uniform(-1e4, 1e4, len(sats))
    for i, sat in enumerate(sats):
        los = _los(rx, sat)
        corr = float(np.dot(los, delta_rx) + delta_cb)
        d_cycles = corr / LAM
        prev.append(_Meas(0, i + 1, sat, base_cycles[i]))
        cur.append(_Meas(0, i + 1, sat, base_cycles[i] + d_cycles))

    vel = estimate_velocity_from_tdcp(rx, prev, cur, dt=dt, wavelength=LAM)
    assert vel is not None
    assert np.allclose(vel, delta_rx / dt, rtol=1e-5, atol=1e-5)


def test_tdcp_supports_negative_carrier_phase_sign():
    rng = np.random.default_rng(10)
    rx = np.array([1.0e6, 2.0e6, 3.0e6])
    dt = 1.0
    delta_rx = np.array([2.5, -1.5, 0.7])
    delta_cb = -0.4

    sats = [
        rx + np.array([20e6, 0.0, 0.0]),
        rx + np.array([0.0, 21e6, 0.0]),
        rx + np.array([0.0, 0.0, 22e6]),
        rx + np.array([15e6, 15e6, 10e6]),
    ]

    prev: list[_Meas] = []
    cur: list[_Meas] = []
    base_cycles = rng.uniform(-1e4, 1e4, len(sats))
    for i, sat in enumerate(sats):
        los = _los(rx, sat)
        corr = float(np.dot(los, delta_rx) + delta_cb)
        d_cycles = -corr / LAM
        prev.append(_Meas(0, i + 1, sat, base_cycles[i]))
        cur.append(_Meas(0, i + 1, sat, base_cycles[i] + d_cycles))

    vel = estimate_velocity_from_tdcp(
        rx,
        prev,
        cur,
        dt=dt,
        wavelength=LAM,
        carrier_phase_sign=-1.0,
    )

    assert vel is not None
    assert np.allclose(vel, delta_rx / dt, rtol=1e-5, atol=1e-5)


def test_tdcp_supports_negative_receiver_motion_sign():
    rng = np.random.default_rng(11)
    rx = np.array([1.0e6, 2.0e6, 3.0e6])
    dt = 1.0
    delta_rx = np.array([2.5, -1.5, 0.7])
    delta_cb = -0.4

    sats = [
        rx + np.array([20e6, 0.0, 0.0]),
        rx + np.array([0.0, 21e6, 0.0]),
        rx + np.array([0.0, 0.0, 22e6]),
        rx + np.array([15e6, 15e6, 10e6]),
    ]

    prev: list[_Meas] = []
    cur: list[_Meas] = []
    base_cycles = rng.uniform(-1e4, 1e4, len(sats))
    for i, sat in enumerate(sats):
        los = _los(rx, sat)
        corr = float(-np.dot(los, delta_rx) + delta_cb)
        d_cycles = corr / LAM
        prev.append(_Meas(0, i + 1, sat, base_cycles[i]))
        cur.append(_Meas(0, i + 1, sat, base_cycles[i] + d_cycles))

    vel = estimate_velocity_from_tdcp(
        rx,
        prev,
        cur,
        dt=dt,
        wavelength=LAM,
        receiver_motion_sign=-1.0,
    )

    assert vel is not None
    assert np.allclose(vel, delta_rx / dt, rtol=1e-5, atol=1e-5)


def test_tdcp_with_satellite_motion_and_clock():
    """All satellites share the same ECEF velocity; include sat clock drift."""
    rx = np.array([1.0e6, 2.0e6, 3.0e6])
    dt = 1.0
    vel_true = np.array([1.0, -0.5, 0.25])
    delta_rx = vel_true * dt
    delta_cb = 0.0
    sat_clock_drift = 1e-9
    sat_vel = np.array([100.0, -50.0, 30.0])

    offsets = [
        np.array([20e6, 0.0, 0.0]),
        np.array([0.0, 21e6, 0.0]),
        np.array([0.0, 0.0, 22e6]),
        np.array([15e6, 15e6, 10e6]),
    ]
    prev: list[_Meas] = []
    cur: list[_Meas] = []
    for i, off in enumerate(offsets):
        sp = rx + off
        sc = sp + sat_vel * dt
        los = _los(rx, 0.5 * (sp + sc))
        v_avg = sat_vel
        sat_range_change = float(np.dot(los, v_avg) * dt)
        drift_avg = sat_clock_drift
        sat_clock_change = drift_avg * C_LIGHT * dt
        d_l_m = (
            float(np.dot(los, delta_rx))
            + delta_cb
            + sat_range_change
            - sat_clock_change
        )
        prev.append(_Meas(0, i + 1, sp, 5000.0 + i, sat_vel, sat_clock_drift))
        cur.append(
            _Meas(0, i + 1, sc, 5000.0 + i + d_l_m / LAM, sat_vel, sat_clock_drift)
        )

    vel = estimate_velocity_from_tdcp(rx, prev, cur, dt=dt, wavelength=LAM, min_sats=4)
    assert vel is not None
    assert np.allclose(vel, vel_true, rtol=1e-5, atol=1e-5)


def test_tdcp_insufficient_sats_returns_none():
    rx = np.array([1e6, 2e6, 3e6])
    prev = [
        _Meas(0, 1, rx + np.array([20e6, 0, 0]), 0.0),
        _Meas(0, 2, rx + np.array([0, 20e6, 0]), 0.0),
    ]
    cur = [
        _Meas(0, 1, prev[0].satellite_ecef, 0.1),
        _Meas(0, 2, prev[1].satellite_ecef, 0.1),
    ]
    assert estimate_velocity_from_tdcp(rx, prev, cur, dt=1.0, min_sats=4) is None


def test_l1_wavelength_alias():
    assert L1_WAVELENGTH == pytest.approx(LAM)


def test_one_row_per_satellite_prefers_higher_snr():
    """Regression: dict-last-wins used to mix L1/L2 for the same PRN (UrbanNav)."""
    a = _Meas(0, 5, np.array([1e7, 0.0, 0.0]), 1.0, snr=35.0)
    b = _Meas(0, 5, np.array([1e7, 1.0, 0.0]), 2.0, snr=48.0)
    c = _Meas(0, 6, np.array([0.0, 2e7, 0.0]), 3.0, snr=40.0)
    picked = _one_row_per_satellite([a, b, c])
    assert picked[(0, 5)] is b
    assert picked[(0, 6)] is c


def test_one_row_omits_snr_tie():
    dup_a = _Meas(0, 5, np.array([1e7, 0.0, 0.0]), 1.0, snr=40.0)
    dup_b = _Meas(0, 5, np.array([1e7, 0.0, 0.0]), 9.0, snr=40.0)
    ok = _Meas(0, 6, np.array([0.0, 2e7, 0.0]), 3.0, snr=41.0)
    picked = _one_row_per_satellite([dup_a, dup_b, ok])
    assert (0, 5) not in picked
    assert picked[(0, 6)] is ok


def test_tdcp_duplicate_prn_uses_high_snr_row():
    """Two frequencies same PRN: low-SNR row has bogus carrier; high-SNR is consistent."""
    rng = np.random.default_rng(42)
    rx = np.array([1.0e6, 2.0e6, 3.0e6])
    dt = 1.0
    delta_rx = np.array([2.0, -1.0, 0.5])
    delta_cb = 0.5

    sats = [
        rx + np.array([20e6, 0.0, 0.0]),
        rx + np.array([0.0, 21e6, 0.0]),
        rx + np.array([0.0, 0.0, 22e6]),
        rx + np.array([15e6, 15e6, 10e6]),
    ]
    bogus_L2_scale = LAM / 0.24

    prev: list[_Meas] = []
    cur: list[_Meas] = []
    base = rng.uniform(-5e3, 5e3, len(sats))
    for i, sat in enumerate(sats):
        los = _los(rx, sat)
        corr = float(np.dot(los, delta_rx) + delta_cb)
        d_cycles = corr / LAM
        prn = i + 1
        prev.extend(
            [
                _Meas(0, prn, sat, base[i], snr=42.0),
                _Meas(
                    0,
                    prn,
                    sat,
                    base[i] + 1e6,
                    snr=30.0,
                ),
            ]
        )
        cur.extend(
            [
                _Meas(0, prn, sat, base[i] + d_cycles, snr=42.0),
                _Meas(
                    0,
                    prn,
                    sat,
                    base[i] + 1e6 + d_cycles * (bogus_L2_scale / LAM),
                    snr=30.0,
                ),
            ]
        )

    vel = estimate_velocity_from_tdcp(rx, prev, cur, dt=dt, wavelength=LAM)
    assert vel is not None
    assert np.allclose(vel, delta_rx / dt, rtol=1e-4, atol=1e-4)


def test_tdcp_high_postfit_rms_returns_none():
    """Corrupt one link: residuals blow up → reject (max_postfit_rms_m)."""
    rng = np.random.default_rng(1)
    rx = np.array([1.0e6, 2.0e6, 3.0e6])
    dt = 1.0
    delta_rx = np.array([1.0, 1.0, 1.0])
    delta_cb = 0.0

    sats = [
        rx + np.array([20e6, 0.0, 0.0]),
        rx + np.array([0.0, 21e6, 0.0]),
        rx + np.array([0.0, 0.0, 22e6]),
        rx + np.array([18e6, 12e6, 5e6]),
    ]

    prev: list[_Meas] = []
    cur: list[_Meas] = []
    base = rng.uniform(-1e4, 1e4, len(sats))
    for i, sat in enumerate(sats):
        los = _los(rx, sat)
        corr = float(np.dot(los, delta_rx) + delta_cb)
        d_cycles = corr / LAM
        prev.append(_Meas(0, i + 1, sat, base[i]))
        dc = d_cycles + (8000.0 if i == 2 else 0.0)
        cur.append(_Meas(0, i + 1, sat, base[i] + dc))

    assert estimate_velocity_from_tdcp(
        rx, prev, cur, dt=dt, wavelength=LAM, max_postfit_rms_m=40.0
    ) is None


def test_tdcp_excessive_speed_returns_none():
    rx = np.array([1.0e6, 2.0e6, 3.0e6])
    dt = 0.1
    delta_rx = np.array([600.0, -50.0, 25.0])
    delta_cb = 0.0
    sats = [
        rx + np.array([20e6, 0.0, 0.0]),
        rx + np.array([0.0, 21e6, 0.0]),
        rx + np.array([0.0, 0.0, 22e6]),
        rx + np.array([15e6, 15e6, 10e6]),
    ]

    prev: list[_Meas] = []
    cur: list[_Meas] = []
    for i, s in enumerate(sats):
        los = _los(rx, s)
        corr = float(np.dot(los, delta_rx) + delta_cb)
        prev.append(_Meas(0, i + 1, s, 1000.0 + i))
        cur.append(_Meas(0, i + 1, s, 1000.0 + i + corr / LAM))

    assert estimate_velocity_from_tdcp(rx, prev, cur, dt=dt, wavelength=LAM) is None

    vel = estimate_velocity_from_tdcp(
        rx,
        prev,
        cur,
        dt=dt,
        wavelength=LAM,
        max_velocity_mps=10000.0,
    )
    assert vel is not None
    assert np.allclose(vel, delta_rx / dt, rtol=1e-5, atol=1e-5)


def test_estimate_velocity_from_tdcp_with_metrics():
    rx = np.array([1.0e6, 2.0e6, 3.0e6])
    dt = 1.0
    delta_rx = np.array([1.0, 0.0, 0.0])
    delta_cb = 0.0
    sats = [
        rx + np.array([20e6, 0.0, 0.0]),
        rx + np.array([0.0, 21e6, 0.0]),
        rx + np.array([0.0, 0.0, 22e6]),
        rx + np.array([12e6, 12e6, 9e6]),
    ]
    prev: list[_Meas] = []
    cur: list[_Meas] = []
    for i, sat in enumerate(sats):
        los = _los(rx, sat)
        dcy = (float(np.dot(los, delta_rx) + delta_cb)) / LAM
        prev.append(_Meas(0, i + 1, sat, 1000.0 + i))
        cur.append(_Meas(0, i + 1, sat, 1000.0 + i + dcy))
    v, rms = estimate_velocity_from_tdcp_with_metrics(rx, prev, cur, dt=dt, wavelength=LAM)
    assert v is not None
    assert np.allclose(v, delta_rx / dt, rtol=1e-5)
    assert rms < 1.0
    v2, rms2 = estimate_velocity_from_tdcp_with_metrics(rx, prev, cur, dt=-0.5)
    assert v2 is None
    assert np.isnan(rms2)


def test_tdcp_dt_nonpositive_returns_none():
    rx = np.array([1e6, 2e6, 3e6])
    p = [_Meas(0, 1, rx + np.array([20e6, 0, 0]), 1.0)]
    c = [_Meas(0, 1, p[0].satellite_ecef, 1.1)]
    assert estimate_velocity_from_tdcp(rx, p, c, dt=0.0) is None
    assert estimate_velocity_from_tdcp(rx, p, c, dt=-1.0) is None
    assert estimate_velocity_from_tdcp(rx, [], c, dt=1.0) is None


def test_tdcp_elevation_weight_same_velocity_if_all_equal_elevation():
    """Uniform sin²(el) scales all rows equally → same LS velocity."""
    rng = np.random.default_rng(1)
    rx = np.array([1.0e6, 2.0e6, 3.0e6])
    dt = 1.0
    delta_rx = np.array([5.1, -2.3, 0.8])
    delta_cb = 1.5
    el = 0.7

    sats = [
        rx + np.array([20e6, 0.0, 0.0]),
        rx + np.array([0.0, 21e6, 0.0]),
        rx + np.array([0.0, 0.0, 22e6]),
        rx + np.array([15e6, 15e6, 10e6]),
    ]
    prev: list[_Meas] = []
    cur: list[_Meas] = []
    base_cycles = rng.uniform(-1e4, 1e4, len(sats))
    for i, sat in enumerate(sats):
        los = _los(rx, sat)
        corr = float(np.dot(los, delta_rx) + delta_cb)
        d_cycles = corr / LAM
        prev.append(_Meas(0, i + 1, sat, base_cycles[i], elevation=el))
        cur.append(_Meas(0, i + 1, sat, base_cycles[i] + d_cycles, elevation=el))

    v0 = estimate_velocity_from_tdcp(rx, prev, cur, dt=dt, wavelength=LAM, elevation_weight=False)
    v1 = estimate_velocity_from_tdcp(rx, prev, cur, dt=dt, wavelength=LAM, elevation_weight=True)
    assert v0 is not None and v1 is not None
    assert np.allclose(v0, v1, rtol=1e-6, atol=1e-8)


def test_tdcp_elevation_weight_closer_to_truth_when_low_elevation_row_is_bad():
    """One contaminated row at low elevation; weighting should reduce bias."""
    rx = np.array([1.0e6, 2.0e6, 3.0e6])
    dt = 1.0
    delta_rx = np.array([3.0, -1.0, 0.5])
    delta_cb = 0.0
    el_hi = float(np.pi / 2)
    el_lo = 0.18

    sats = [
        rx + np.array([20e6, 0.0, 0.0]),
        rx + np.array([0.0, 21e6, 0.0]),
        rx + np.array([0.0, 0.0, 22e6]),
        rx + np.array([15e6, 15e6, 10e6]),
        rx + np.array([10e6, 10e6, 20e6]),
    ]
    prev: list[_Meas] = []
    cur: list[_Meas] = []
    # Range error on one row (m); keep moderate so the unweighted fit is still accepted.
    contam = 28.0
    for i, sat in enumerate(sats):
        los = _los(rx, sat)
        dcy = (float(np.dot(los, delta_rx) + delta_cb)) / LAM
        if i == 0:
            dcy += contam / LAM
        el = el_lo if i == 0 else el_hi
        prev.append(_Meas(0, i + 1, sat, 1000.0 + i, elevation=el))
        cur.append(_Meas(0, i + 1, sat, 1000.0 + i + dcy, elevation=el))

    true_vel = delta_rx / dt
    v_u = estimate_velocity_from_tdcp(
        rx,
        prev,
        cur,
        dt=dt,
        wavelength=LAM,
        max_postfit_rms_m=80.0,
        elevation_weight=False,
    )
    v_w = estimate_velocity_from_tdcp(
        rx,
        prev,
        cur,
        dt=dt,
        wavelength=LAM,
        max_postfit_rms_m=80.0,
        elevation_weight=True,
    )
    assert v_u is not None and v_w is not None
    err_u = float(np.linalg.norm(v_u - true_vel))
    err_w = float(np.linalg.norm(v_w - true_vel))
    assert err_w < err_u
