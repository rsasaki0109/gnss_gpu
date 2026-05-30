"""Unit tests for the GSDC2023 TDCP error-state correction smoother math.

These cover the deterministic numerical core of
:mod:`experiments.eval_gsdc2023_tdcp_correction_smoother` — the tridiagonal
solver, the per-axis correction least-squares, the ENU rotation, the
ECEF<->LLA round trip, and the quality gating inside the smoother.  They do not
touch the GNSS data pipeline (``build_trip_arrays``), so they run without the
GSDC2023 dataset.
"""

from __future__ import annotations

import numpy as np

from experiments.eval_gsdc2023_tdcp_correction_smoother import (
    TdcpSmootherConfig,
    _apply_tdcp_smoother,
    _ecef_to_enu_delta,
    _ecef_to_lla,
    _lla_to_ecef,
    _solve_correction_axis,
    _solve_tridiagonal,
)


def _dense_tridiag(diag: np.ndarray, off: np.ndarray) -> np.ndarray:
    n = diag.size
    m = np.diag(diag.astype(np.float64))
    for i in range(n - 1):
        m[i, i + 1] = off[i]
        m[i + 1, i] = off[i]
    return m


def test_solve_tridiagonal_matches_dense_solve():
    diag = np.array([4.0, 5.0, 6.0, 7.0])
    off = np.array([1.0, -2.0, 0.5])
    rhs = np.array([1.0, 2.0, 3.0, 4.0])
    got = _solve_tridiagonal(diag, off, rhs)
    want = np.linalg.solve(_dense_tridiag(diag, off), rhs)
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9)


def test_solve_tridiagonal_empty():
    assert _solve_tridiagonal(np.empty(0), np.empty(0), np.empty(0)).size == 0


def test_solve_tridiagonal_single():
    got = _solve_tridiagonal(np.array([3.0]), np.empty(0), np.array([6.0]))
    np.testing.assert_allclose(got, [2.0], rtol=1e-12)


def _cfg(**kw) -> TdcpSmootherConfig:
    base = dict(
        label="t",
        sigma_anchor_m=1e6,  # weak anchor so TDCP dominates
        sigma_tdcp_m=0.01,
        max_condition=1e9,
        max_postfit_rms_m=1e9,
        min_pairs=0,
        max_delta_m=1e9,
    )
    base.update(kw)
    return TdcpSmootherConfig(**base)


def test_correction_axis_integrates_perfect_deltas():
    # With a near-zero anchor weight, the correction trajectory should be the
    # cumulative sum of the interval deltas (up to a constant the anchor pins
    # toward zero).  Pin the constant by checking differences.
    deltas = np.array([0.5, -0.2, 1.0, 0.3])
    valid = np.ones(deltas.size, dtype=bool)
    corr = _solve_correction_axis(deltas, valid, _cfg())
    np.testing.assert_allclose(np.diff(corr), deltas, rtol=1e-4, atol=1e-4)


def test_correction_axis_anchor_pulls_to_zero():
    # A strong anchor and weak TDCP should keep corrections near zero.
    deltas = np.array([5.0, 5.0, 5.0])
    valid = np.ones(deltas.size, dtype=bool)
    corr = _solve_correction_axis(deltas, valid, _cfg(sigma_anchor_m=0.01, sigma_tdcp_m=1e6))
    assert np.max(np.abs(corr)) < 1e-2


def test_correction_axis_skips_invalid_intervals():
    deltas = np.array([1.0, 999.0, 1.0])
    valid = np.array([True, False, True])
    corr = _solve_correction_axis(deltas, valid, _cfg())
    # The invalid middle interval contributes no coupling, so the chain splits
    # into two independent segments {c0,c1} and {c2,c3}.  Each valid interval
    # still reproduces its delta as the step across it; the huge invalid delta
    # never reaches the trajectory.
    np.testing.assert_allclose(corr[1] - corr[0], 1.0, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(corr[3] - corr[2], 1.0, rtol=1e-4, atol=1e-4)
    assert np.max(np.abs(corr)) < 5.0  # 999 m delta did not leak in


def test_correction_axis_clamps_max_delta():
    deltas = np.array([100.0, 100.0])
    valid = np.ones(2, dtype=bool)
    corr = _solve_correction_axis(deltas, valid, _cfg(max_delta_m=2.0))
    # Each delta clamped to 2.0, so the trajectory step is ~2.0 not 100.0.
    assert np.all(np.abs(np.diff(corr)) <= 2.0 + 1e-6)


def test_ecef_lla_round_trip():
    lat = np.array([35.681, -33.868, 0.0])
    lon = np.array([139.767, 151.209, -120.0])
    h = np.array([40.0, 58.0, 1000.0])
    xyz = _lla_to_ecef(lat, lon, h)
    lat2, lon2, h2 = _ecef_to_lla(xyz)
    np.testing.assert_allclose(lat2, lat, atol=1e-7)
    np.testing.assert_allclose(lon2, lon, atol=1e-7)
    np.testing.assert_allclose(h2, h, atol=1e-3)


def test_ecef_to_enu_delta_axes():
    # At the equator/prime meridian the ECEF axes map cleanly to ENU:
    # +Y(ECEF)->East, +Z(ECEF)->North, +X(ECEF)->Up.
    d = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    enu = _ecef_to_enu_delta(d, 0.0, 0.0)
    np.testing.assert_allclose(enu[0], [1.0, 0.0, 0.0], atol=1e-9)  # +Y -> East
    np.testing.assert_allclose(enu[1], [0.0, 1.0, 0.0], atol=1e-9)  # +Z -> North
    np.testing.assert_allclose(enu[2], [0.0, 0.0, 1.0], atol=1e-9)  # +X -> Up


def test_apply_smoother_quality_gate_rejects_bad_intervals():
    # Two intervals; the second fails the postfit-RMS gate and must be ignored,
    # leaving only the first interval's correction influencing the trajectory.
    lat = np.array([35.0, 35.0, 35.0])
    lon = np.array([139.0, 139.0, 139.0])
    xyz = _lla_to_ecef(lat, lon, np.zeros(3))
    # dpos in ECEF that maps to ~1 m East steps; supply via ENU-equivalent.
    dpos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    interval_valid = np.array([True, True])
    quality = [
        {"pair_count": 10.0, "postfit_rms_m": 0.01, "condition_number": 1.0},
        {"pair_count": 10.0, "postfit_rms_m": 99.0, "condition_number": 1.0},  # bad RMS
    ]
    cfg = _cfg(max_postfit_rms_m=0.05, min_pairs=8, max_condition=15.0)
    lat_c, lon_c, valid_count = _apply_tdcp_smoother(
        lat, lon, xyz, dpos, interval_valid, quality, cfg
    )
    assert valid_count == 1
    assert lat_c.shape == lat.shape and lon_c.shape == lon.shape
    assert np.all(np.isfinite(lat_c)) and np.all(np.isfinite(lon_c))
