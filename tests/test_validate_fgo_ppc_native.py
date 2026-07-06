"""Fast tests for validate_fgo_ppc native/in-repo Doppler plumbing."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from validate_fgo_ppc import (  # noqa: E402
    C_LIGHT as C_LIGHT_TEST,
    DEFAULT_CONSTELLATION_SIGMA_SCALE,
    _apply_constellation_sigma_scaling,
    _apply_elevation_mask,
    _chunk_ranges,
    _doppler_hz_to_range_rate,
    _elevation_deg_per_epoch,
    _gate_doppler_outliers_per_epoch,
    _per_satellite_wavelength_m,
    _seed_chunk_boundary_state,
    _solve_fgo_vd_chunked,
    _vd_state_stride,
    export_trajectory_csv,
    run_fgo_on_ppc_native,
)


def test_chunk_ranges_splits_long_sequences():
    assert _chunk_ranges(100, 200) == [(0, 100)]
    assert _chunk_ranges(4500, 1000) == [(0, 1000), (1000, 2000), (2000, 3000), (3000, 4000), (4000, 4500)]


def test_vd_state_stride_matches_native_clock_layout():
    assert _vd_state_stride(1) == 8


def test_seed_chunk_boundary_state_copies_kinematic_block():
    seg = np.zeros((3, 8), dtype=np.float64)
    prev = np.arange(8, dtype=np.float64)
    _seed_chunk_boundary_state(seg, prev)
    np.testing.assert_array_equal(seg[0], prev)


def test_solve_fgo_vd_chunked_records_per_chunk_stats(monkeypatch):
    calls: list[int] = []

    def _fake_fgo_gnss_lm_vd(sat_ecef, pseudorange, weights, state, **kwargs):
        calls.append(int(sat_ecef.shape[0]))
        state[:, :3] += 0.01
        return 2, 12.5

    import validate_fgo_ppc as mod

    monkeypatch.setattr(mod, "fgo_gnss_lm_vd", _fake_fgo_gnss_lm_vd)

    n_epoch, n_sat = 5, 4
    sat_ecef = np.ones((n_epoch, n_sat, 3), dtype=np.float64)
    pseudorange = np.full((n_epoch, n_sat), 2.0e7, dtype=np.float64)
    weights = np.ones((n_epoch, n_sat), dtype=np.float64)
    state = np.zeros((n_epoch, 8), dtype=np.float64)
    dt = np.full(n_epoch, 0.2, dtype=np.float64)

    total_iters, mean_mse, chunk_stats = _solve_fgo_vd_chunked(
        sat_ecef,
        pseudorange,
        weights,
        state,
        n_clock=1,
        motion_sigma_m=1.0,
        clock_drift_sigma_m=1.0,
        fgo_iters=2,
        sat_vel=None,
        doppler=None,
        doppler_weights=None,
        sat_clock_drift=None,
        dt=dt,
        chunk_epochs=2,
    )

    assert calls == [2, 2, 1]
    assert total_iters == 6
    assert mean_mse == pytest.approx(12.5)
    assert len(chunk_stats) == 3
    assert all(chunk["status"] == "ok" for chunk in chunk_stats)
    assert chunk_stats[1]["start"] == 2
    assert state[2, 0] == pytest.approx(0.02)


def test_solve_fgo_vd_chunked_rejects_oversized_chunk_before_native(monkeypatch):
    def _should_not_run(*_args, **_kwargs):
        raise AssertionError("native solver must not be called when n_state exceeds cap")

    import validate_fgo_ppc as mod

    monkeypatch.setattr(mod, "fgo_gnss_lm_vd", _should_not_run)

    n_epoch, n_sat = 3000, 4
    sat_ecef = np.ones((n_epoch, n_sat, 3), dtype=np.float64)
    pseudorange = np.full((n_epoch, n_sat), 2.0e7, dtype=np.float64)
    weights = np.ones((n_epoch, n_sat), dtype=np.float64)
    state = np.zeros((n_epoch, 8), dtype=np.float64)
    dt = np.full(n_epoch, 0.2, dtype=np.float64)

    total_iters, mean_mse, chunk_stats = _solve_fgo_vd_chunked(
        sat_ecef,
        pseudorange,
        weights,
        state,
        n_clock=1,
        motion_sigma_m=1.0,
        clock_drift_sigma_m=1.0,
        fgo_iters=2,
        sat_vel=None,
        doppler=None,
        doppler_weights=None,
        sat_clock_drift=None,
        dt=dt,
        chunk_epochs=3000,
    )

    assert total_iters == -1
    assert mean_mse == 0.0
    assert chunk_stats[0]["status"] == "n_state_cap"


def test_doppler_hz_to_range_rate_sign_and_units():
    hz = np.array([100.0, 0.0, np.nan], dtype=np.float64)
    rr = _doppler_hz_to_range_rate(hz, wavelength_m=0.19)
    assert rr[0] == pytest.approx(-19.0)
    assert rr[1] == 0.0
    assert rr[2] == 0.0


def test_export_trajectory_csv_columns(tmp_path: Path):
    times = np.array([100.0, 100.2], dtype=np.float64)
    ecef = np.array(
        [
            [-3958080.0, 3350070.0, 3700660.0],
            [-3958081.0, 3350071.0, 3700661.0],
        ],
        dtype=np.float64,
    )
    out = tmp_path / "traj.csv"
    export_trajectory_csv(out, times, ecef)
    with out.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert set(rows[0]) == {
        "tow",
        "lat_deg",
        "lon_deg",
        "height_m",
        "ecef_x",
        "ecef_y",
        "ecef_z",
        "fix",
    }
    assert float(rows[0]["tow"]) == pytest.approx(100.0)


def test_run_fgo_on_ppc_native_rejects_rtklib_mode(tmp_path: Path):
    run_dir = tmp_path / "tokyo" / "run1"
    run_dir.mkdir(parents=True)
    for name in ("rover.obs", "base.obs", "base.nav", "reference.csv"):
        (run_dir / name).write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="off\\|in-repo"):
        run_fgo_on_ppc_native(run_dir, doppler_mode="rtklib")


# --- D2: multi-clock chunk boundary seeding -------------------------------


def test_seed_chunk_boundary_state_multi_clock_carries_all_clocks_and_drift():
    n_clock = 5
    seg = np.zeros((3, 7 + n_clock), dtype=np.float64)
    prev = np.arange(7 + n_clock, dtype=np.float64)
    _seed_chunk_boundary_state(seg, prev, n_clock=n_clock)
    np.testing.assert_array_equal(seg[0], prev)
    # Only the first (pinned) row is touched.
    np.testing.assert_array_equal(seg[1], np.zeros(7 + n_clock))


def test_seed_chunk_boundary_state_default_n_clock_matches_legacy_behaviour():
    seg = np.zeros((3, 8), dtype=np.float64)
    prev = np.arange(8, dtype=np.float64)
    _seed_chunk_boundary_state(seg, prev)
    np.testing.assert_array_equal(seg[0], prev)


# --- D1: per-satellite Doppler wavelength + robust outlier gating ---------


def test_per_satellite_wavelength_m_maps_known_systems_and_nans_glonass():
    used_prns = [["G01", "R02", "E03", "C04", "J05"]]
    wl = _per_satellite_wavelength_m(used_prns, max_sats=5)
    assert wl.shape == (1, 5)
    assert wl[0, 0] == pytest.approx(C_LIGHT_TEST / 1575.42e6)  # G
    assert np.isnan(wl[0, 1])  # R (GLONASS FDMA, unsupported)
    assert wl[0, 2] == pytest.approx(C_LIGHT_TEST / 1575.42e6)  # E
    assert wl[0, 3] == pytest.approx(C_LIGHT_TEST / 1561.098e6)  # C
    assert wl[0, 4] == pytest.approx(C_LIGHT_TEST / 1575.42e6)  # J


def test_doppler_hz_to_range_rate_accepts_per_satellite_wavelength_array():
    hz = np.array([100.0, 100.0, 100.0], dtype=np.float64)
    wl = np.array([0.19, np.nan, 0.0], dtype=np.float64)
    rr = _doppler_hz_to_range_rate(hz, wavelength_m=wl)
    assert rr[0] == pytest.approx(-19.0)
    assert rr[1] == 0.0  # nan wavelength (GLONASS) -> dropped
    assert rr[2] == 0.0  # zero wavelength -> dropped


def _synthetic_doppler_epoch(n_sat: int = 6):
    """4-satellite well-conditioned geometry + a clean receiver velocity/drift,
    used to build noise-free Doppler observations for gating tests."""
    rx_pos = np.array([-3958080.0, 3350070.0, 3700660.0])
    rng = np.random.default_rng(0)
    directions = rng.normal(size=(n_sat, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    sat_ecef = rx_pos[None, :] + directions * 2.2e7
    sat_vel = rng.normal(scale=500.0, size=(n_sat, 3))
    rx_vel_true = np.array([5.0, -3.0, 1.0])
    drift_true = 2.0
    los = (sat_ecef - rx_pos[None, :])
    los /= np.linalg.norm(los, axis=1, keepdims=True)
    rr_clean = drift_true + np.einsum("ij,ij->i", los, sat_vel - rx_vel_true[None, :])
    return sat_ecef, sat_vel, rr_clean


def test_gate_doppler_outliers_per_epoch_rejects_single_bad_satellite():
    sat_ecef, sat_vel, rr_clean = _synthetic_doppler_epoch(n_sat=8)
    rr_obs = rr_clean.copy()
    rr_obs[3] += 500.0  # inject a gross outlier (e.g. multipath/cycle-slip spike)

    sat_ecef_b = sat_ecef[None, :, :]
    sat_vel_b = sat_vel[None, :, :]
    rr_obs_b = rr_obs[None, :]
    weights = np.ones((1, 8), dtype=np.float64)
    sat_clock_drift = np.zeros((1, 8), dtype=np.float64)
    rx_state = np.zeros((1, 8), dtype=np.float64)
    rx_state[0, :3] = [-3958080.0, 3350070.0, 3700660.0]

    gated_weights, stats = _gate_doppler_outliers_per_epoch(
        sat_ecef_b, sat_vel_b, sat_clock_drift, rr_obs_b, weights, rx_state,
        gate_sigma=3.0, min_sigma_mps=1.0,
    )

    assert gated_weights[0, 3] == 0.0
    assert stats["n_obs_gated"] == 1
    assert stats["n_epochs_gated"] == 1
    kept = [i for i in range(8) if i != 3]
    assert np.all(gated_weights[0, kept] > 0.0)


def test_gate_doppler_outliers_per_epoch_leaves_clean_epoch_untouched():
    sat_ecef, sat_vel, rr_clean = _synthetic_doppler_epoch(n_sat=8)
    sat_ecef_b = sat_ecef[None, :, :]
    sat_vel_b = sat_vel[None, :, :]
    rr_obs_b = rr_clean[None, :]
    weights = np.ones((1, 8), dtype=np.float64)
    sat_clock_drift = np.zeros((1, 8), dtype=np.float64)
    rx_state = np.zeros((1, 8), dtype=np.float64)
    rx_state[0, :3] = [-3958080.0, 3350070.0, 3700660.0]

    gated_weights, stats = _gate_doppler_outliers_per_epoch(
        sat_ecef_b, sat_vel_b, sat_clock_drift, rr_obs_b, weights, rx_state,
        gate_sigma=3.0, min_sigma_mps=1.0,
    )

    assert stats["n_obs_gated"] == 0
    np.testing.assert_array_equal(gated_weights, weights)


def test_gate_doppler_outliers_per_epoch_skips_epochs_with_too_few_obs():
    sat_ecef, sat_vel, rr_clean = _synthetic_doppler_epoch(n_sat=8)
    sat_ecef_b = sat_ecef[None, :, :]
    sat_vel_b = sat_vel[None, :, :]
    rr_obs_b = rr_clean[None, :]
    weights = np.zeros((1, 8), dtype=np.float64)
    weights[0, :3] = 1.0  # below min_obs=5
    sat_clock_drift = np.zeros((1, 8), dtype=np.float64)
    rx_state = np.zeros((1, 8), dtype=np.float64)
    rx_state[0, :3] = [-3958080.0, 3350070.0, 3700660.0]

    gated_weights, stats = _gate_doppler_outliers_per_epoch(
        sat_ecef_b, sat_vel_b, sat_clock_drift, rr_obs_b, weights, rx_state,
        gate_sigma=3.0, min_obs=5,
    )

    assert stats["n_epochs_gated"] == 0
    np.testing.assert_array_equal(gated_weights, weights)


# --- D1: doppler_huber_k / sys_kind plumbed through the chunked VD solver -


def test_solve_fgo_vd_chunked_passes_doppler_huber_k_and_sys_kind(monkeypatch):
    captured: list[dict] = []

    def _fake_fgo_gnss_lm_vd(sat_ecef, pseudorange, weights, state, **kwargs):
        captured.append(kwargs)
        return 1, 0.0

    import validate_fgo_ppc as mod

    monkeypatch.setattr(mod, "fgo_gnss_lm_vd", _fake_fgo_gnss_lm_vd)

    n_epoch, n_sat, n_clock = 4, 4, 2
    sat_ecef = np.ones((n_epoch, n_sat, 3), dtype=np.float64)
    pseudorange = np.full((n_epoch, n_sat), 2.0e7, dtype=np.float64)
    weights = np.ones((n_epoch, n_sat), dtype=np.float64)
    state = np.zeros((n_epoch, 7 + n_clock), dtype=np.float64)
    dt = np.full(n_epoch, 0.2, dtype=np.float64)
    sys_kind = np.zeros((n_epoch, n_sat), dtype=np.int32)
    sys_kind[:, 2:] = 1

    _solve_fgo_vd_chunked(
        sat_ecef, pseudorange, weights, state,
        n_clock=n_clock, motion_sigma_m=1.0, clock_drift_sigma_m=1.0,
        fgo_iters=2, sat_vel=None, doppler=None, doppler_weights=None,
        sat_clock_drift=None, dt=dt, sys_kind=sys_kind, doppler_huber_k=7.5,
    )

    assert len(captured) == 1
    assert captured[0]["doppler_huber_k"] == pytest.approx(7.5)
    np.testing.assert_array_equal(captured[0]["sys_kind"], sys_kind)


# --- D3: IMU preintegration segment slicing inside the chunked VD solver --


def test_solve_fgo_vd_chunked_slices_imu_deltas_per_chunk(monkeypatch):
    from experiments.gsdc2023_imu import IMUPreintegration

    captured: list[dict] = []

    def _fake_fgo_gnss_lm_vd(sat_ecef, pseudorange, weights, state, **kwargs):
        captured.append(kwargs)
        return 1, 0.0

    import validate_fgo_ppc as mod

    monkeypatch.setattr(mod, "fgo_gnss_lm_vd", _fake_fgo_gnss_lm_vd)

    n_epoch, n_sat = 5, 4
    sat_ecef = np.ones((n_epoch, n_sat, 3), dtype=np.float64)
    pseudorange = np.full((n_epoch, n_sat), 2.0e7, dtype=np.float64)
    weights = np.ones((n_epoch, n_sat), dtype=np.float64)
    state = np.zeros((n_epoch, 8), dtype=np.float64)
    dt = np.full(n_epoch, 0.2, dtype=np.float64)

    n_interval = n_epoch - 1
    delta_p = np.arange(n_interval * 3, dtype=np.float64).reshape(n_interval, 3)
    delta_v = -delta_p.copy()
    preint = IMUPreintegration(
        epoch_times_ms=np.arange(n_epoch, dtype=np.float64) * 200.0,
        delta_t_s=np.full(n_interval, 0.2),
        delta_v_body=delta_v,
        delta_p_body=delta_p,
        delta_angle_rad=np.zeros((n_interval, 3)),
        sample_count=np.ones(n_interval, dtype=np.int32),
    )

    _solve_fgo_vd_chunked(
        sat_ecef, pseudorange, weights, state,
        n_clock=1, motion_sigma_m=1.0, clock_drift_sigma_m=1.0,
        fgo_iters=2, sat_vel=None, doppler=None, doppler_weights=None,
        sat_clock_drift=None, dt=dt, chunk_epochs=3,
        imu_preint=preint, imu_position_sigma_m=0.5, imu_velocity_sigma_mps=0.3,
    )

    assert len(captured) == 2  # chunks [0:3], [3:5]
    np.testing.assert_allclose(captured[0]["imu_delta_p"], delta_p[0:2])
    np.testing.assert_allclose(captured[0]["imu_delta_v"], delta_v[0:2])
    assert captured[0]["imu_position_sigma_m"] == pytest.approx(0.5)
    assert captured[0]["imu_velocity_sigma_mps"] == pytest.approx(0.3)
    np.testing.assert_allclose(captured[1]["imu_delta_p"], delta_p[3:4])


# --- WP3c work item 1: elevation mask --------------------------------------

# Receiver at lat=0/lon=0 on the WGS84 ellipsoid: local ENU basis is exactly
# (east=+y, north=+z, up=+x), so a satellite placed along
# rx + range * (sin(el), 0, cos(el)) sits at exactly `el` degrees elevation
# (see gnss_gpu.validation.real_residuals.elevation_azimuth's convention).
_RX_EQUATOR_ECEF = np.array([6378137.0, 0.0, 0.0], dtype=np.float64)


def _sat_at_elevation_deg(elevation_deg: float, range_m: float = 2.0e7) -> np.ndarray:
    el_rad = np.radians(elevation_deg)
    direction = np.array([np.sin(el_rad), 0.0, np.cos(el_rad)], dtype=np.float64)
    return _RX_EQUATOR_ECEF + direction * range_m


def test_elevation_deg_per_epoch_matches_known_geometry():
    sat_ecef = np.stack(
        [_sat_at_elevation_deg(30.0), _sat_at_elevation_deg(5.0)],
    )[None, :, :]
    weights = np.ones((1, 2), dtype=np.float64)
    rx_state = np.zeros((1, 3), dtype=np.float64)
    rx_state[0] = _RX_EQUATOR_ECEF

    elev_deg = _elevation_deg_per_epoch(sat_ecef, weights, rx_state)

    assert elev_deg[0, 0] == pytest.approx(30.0, abs=1e-6)
    assert elev_deg[0, 1] == pytest.approx(5.0, abs=1e-6)


def test_elevation_deg_per_epoch_nan_for_unweighted_or_unset_rx():
    sat_ecef = np.stack([_sat_at_elevation_deg(30.0), _sat_at_elevation_deg(5.0)])[None, :, :]
    weights = np.array([[1.0, 0.0]], dtype=np.float64)  # column 1 unweighted
    rx_state = np.zeros((1, 3), dtype=np.float64)  # row 0: rx unset (norm ~ 0)

    elev_deg = _elevation_deg_per_epoch(sat_ecef, weights, rx_state)

    assert np.isnan(elev_deg[0, 1])  # unweighted column always NaN
    assert np.isnan(elev_deg[0, 0])  # rx unset -> whole epoch skipped


def test_apply_elevation_mask_zeroes_low_elevation_satellite():
    sat_ecef = np.stack(
        [_sat_at_elevation_deg(30.0), _sat_at_elevation_deg(5.0)],
    )[None, :, :]
    weights = np.ones((1, 2), dtype=np.float64)
    rx_state = np.zeros((1, 3), dtype=np.float64)
    rx_state[0] = _RX_EQUATOR_ECEF

    out, stats = _apply_elevation_mask(sat_ecef, weights, rx_state, min_elevation_deg=10.0)

    assert out[0, 0] == pytest.approx(1.0)  # 30 deg: kept
    assert out[0, 1] == pytest.approx(0.0)  # 5 deg: masked out
    assert stats["n_obs_masked"] == 1
    assert stats["n_obs_total"] == 2
    assert stats["min_elevation_deg"] == pytest.approx(10.0)


def test_apply_elevation_mask_disabled_returns_weights_unchanged():
    sat_ecef = np.stack([_sat_at_elevation_deg(30.0), _sat_at_elevation_deg(5.0)])[None, :, :]
    weights = np.ones((1, 2), dtype=np.float64)
    rx_state = np.zeros((1, 3), dtype=np.float64)
    rx_state[0] = _RX_EQUATOR_ECEF

    out, stats = _apply_elevation_mask(sat_ecef, weights, rx_state, min_elevation_deg=0.0)

    np.testing.assert_array_equal(out, weights)
    assert stats["n_obs_masked"] == 0


# --- WP3c work item 2b: per-constellation sigma scaling --------------------


def test_default_constellation_sigma_scale_matches_task_e_spec():
    assert DEFAULT_CONSTELLATION_SIGMA_SCALE["G"] == pytest.approx(1.0)
    assert DEFAULT_CONSTELLATION_SIGMA_SCALE["E"] == pytest.approx(1.0)
    assert DEFAULT_CONSTELLATION_SIGMA_SCALE["J"] == pytest.approx(1.0)
    assert DEFAULT_CONSTELLATION_SIGMA_SCALE["C"] == pytest.approx(1.5)
    assert DEFAULT_CONSTELLATION_SIGMA_SCALE["R"] == pytest.approx(2.0)


def test_apply_constellation_sigma_scaling_rescales_by_sigma_squared():
    used_prns = [["G01", "C02", "R03", "E04", "J05"]]
    weights = np.full((1, 5), 4.0, dtype=np.float64)

    out = _apply_constellation_sigma_scaling(weights, used_prns)

    assert out[0, 0] == pytest.approx(4.0)          # G: 1.0x -> unchanged
    assert out[0, 1] == pytest.approx(4.0 / 1.5**2)  # C: 1.5x sigma
    assert out[0, 2] == pytest.approx(4.0 / 2.0**2)  # R: 2.0x sigma
    assert out[0, 3] == pytest.approx(4.0)           # E: unchanged
    assert out[0, 4] == pytest.approx(4.0)           # J: unchanged


def test_apply_constellation_sigma_scaling_honours_custom_overrides():
    used_prns = [["C02"]]
    weights = np.ones((1, 1), dtype=np.float64)

    out = _apply_constellation_sigma_scaling(weights, used_prns, sigma_scale={"C": 3.0})

    assert out[0, 0] == pytest.approx(1.0 / 3.0**2)


def test_apply_constellation_sigma_scaling_skips_already_zero_weights():
    used_prns = [["C02"]]
    weights = np.zeros((1, 1), dtype=np.float64)

    out = _apply_constellation_sigma_scaling(weights, used_prns)

    assert out[0, 0] == pytest.approx(0.0)
