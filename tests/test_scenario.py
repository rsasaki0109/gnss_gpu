"""Tests for the unified scenario engine (gnss_gpu.scenario).

All tests here run without a GPU and without network access: the NAV file is
a small synthetic RINEX 3 fixture (values borrowed from tests/test_ephemeris.py,
which are known-good realistic GPS Kepler parameters), and no PLATEAU mesh is
loaded unless the optional experiments/data fixtures are present on disk.
"""

from __future__ import annotations

import textwrap
import warnings
from pathlib import Path

import numpy as np
import pytest

from gnss_gpu.scenario import EpochRecord, ScenarioConfig, ScenarioResult, run_scenario

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PLATEAU_DIR = _REPO_ROOT / "experiments" / "data" / "plateau_odaiba"
_HAS_PLATEAU_FIXTURE = _PLATEAU_DIR.is_dir()

# ---------------------------------------------------------------------------
# Synthetic RINEX 3 NAV fixtures (GPS G01/G06 values lifted from
# tests/test_ephemeris.py; a synthetic E01 record reuses G01's valid Kepler
# elements under a Galileo system letter so multi-constellation code paths
# have a second system to exercise).
# ---------------------------------------------------------------------------

_NAV_GPS_ONLY = textwrap.dedent("""\
     3.04           N: GNSS NAV DATA    G: GPS              RINEX VERSION / TYPE
    test_program    test_agency         20240115 000000 UTC PGM / RUN BY / DATE
                                                                END OF HEADER
    G01 2024 01 15 00 00 00-3.930553793907E-05-1.023181539495E-12 0.000000000000E+00
         7.800000000000E+01-1.403125000000E+01 4.623016997497E-09 1.245932843990E+00
        -3.464519977570E-06 5.765914916992E-03 7.525086402893E-06 5.153637939453E+03
         5.184000000000E+05-1.303851604462E-07-2.495230285080E-01 5.587935447693E-08
         9.734965789940E-01 2.241562500000E+02 6.859404140730E-01-8.120689826012E-09
         1.132502065007E-10 1.000000000000E+00 2.295000000000E+03 0.000000000000E+00
         2.000000000000E+00 0.000000000000E+00-1.117587089539E-08 7.800000000000E+01
         5.184000000000E+05 4.000000000000E+00
    G06 2024 01 15 02 00 00 1.500000000000E-05 0.000000000000E+00 0.000000000000E+00
         1.000000000000E+02-1.000000000000E+01 4.500000000000E-09 2.000000000000E+00
        -3.000000000000E-06 8.000000000000E-03 7.000000000000E-06 5.153500000000E+03
         5.256000000000E+05-1.000000000000E-07 1.200000000000E+00 5.000000000000E-08
         9.700000000000E-01 2.200000000000E+02-5.000000000000E-01-8.000000000000E-09
         1.000000000000E-10 1.000000000000E+00 2.295000000000E+03 0.000000000000E+00
         2.000000000000E+00 0.000000000000E+00 0.000000000000E+00 1.000000000000E+02
         5.256000000000E+05 4.000000000000E+00
""")

_NAV_MIXED = textwrap.dedent("""\
     3.04           N: GNSS NAV DATA    M: MIXED            RINEX VERSION / TYPE
    test_program    test_agency         20240115 000000 UTC PGM / RUN BY / DATE
                                                                END OF HEADER
    G01 2024 01 15 00 00 00-3.930553793907E-05-1.023181539495E-12 0.000000000000E+00
         7.800000000000E+01-1.403125000000E+01 4.623016997497E-09 1.245932843990E+00
        -3.464519977570E-06 5.765914916992E-03 7.525086402893E-06 5.153637939453E+03
         5.184000000000E+05-1.303851604462E-07-2.495230285080E-01 5.587935447693E-08
         9.734965789940E-01 2.241562500000E+02 6.859404140730E-01-8.120689826012E-09
         1.132502065007E-10 1.000000000000E+00 2.295000000000E+03 0.000000000000E+00
         2.000000000000E+00 0.000000000000E+00-1.117587089539E-08 7.800000000000E+01
         5.184000000000E+05 4.000000000000E+00
    E01 2024 01 15 00 00 00-3.930553793907E-05-1.023181539495E-12 0.000000000000E+00
         7.800000000000E+01-1.403125000000E+01 4.623016997497E-09 1.245932843990E+00
        -3.464519977570E-06 5.765914916992E-03 7.525086402893E-06 5.153637939453E+03
         5.184000000000E+05-1.303851604462E-07-2.495230285080E-01 5.587935447693E-08
         9.734965789940E-01 2.241562500000E+02 6.859404140730E-01-8.120689826012E-09
         1.132502065007E-10 1.000000000000E+00 2.295000000000E+03 0.000000000000E+00
         2.000000000000E+00 0.000000000000E+00-1.117587089539E-08 7.800000000000E+01
         5.184000000000E+05 4.000000000000E+00
""")

# G01's toe (518400.0 s-of-week) falls on 2024-01-15T00:00:00 GPS time -- start
# the scenario there so the ephemeris selection has dt=0 for the primary sat.
_START_TIME = "2024-01-15T00:00:00"
_RX_LAT = 35.6
_RX_LON = 139.7
_RX_ALT = 30.0


def _write_nav(tmp_path: Path, content: str, name: str = "test.nav") -> str:
    nav_file = tmp_path / name
    nav_file.write_text(content)
    return str(nav_file)


def _base_config(nav_file: str, **overrides) -> ScenarioConfig:
    kwargs = dict(
        nav_file=nav_file,
        lat_deg=_RX_LAT,
        lon_deg=_RX_LON,
        alt_m=_RX_ALT,
        start_time=_START_TIME,
        duration_s=2.0,
        step_s=1.0,
        constellations=["G"],
        plateau_dir=None,
        elevation_mask_deg=0.0,
        pr_noise_sigma_zenith_m=0.0,
        pr_noise_sigma_horizon_m=0.0,
        seed=42,
    )
    kwargs.update(overrides)
    return ScenarioConfig(**kwargs)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestSchema:
    def test_epoch_record_has_expected_fields(self, tmp_path):
        nav_file = _write_nav(tmp_path, _NAV_GPS_ONLY)
        result = run_scenario(_base_config(nav_file))

        assert isinstance(result, ScenarioResult)
        assert result.n_epochs == 3  # duration=2s, step=1s -> t=0,1,2

        ep = result.epochs[0]
        assert isinstance(ep, EpochRecord)
        assert isinstance(ep.time_gps_week, int)
        assert isinstance(ep.time_sow, float)
        assert ep.time_utc is not None
        assert ep.rx_ecef.shape == (3,)

        n = ep.n_sat
        assert n >= 1
        for name in (
            "sat_id", "elevation_rad", "azimuth_rad", "is_los",
            "pseudorange_m", "range_geometric_m", "sat_clock_bias_m",
            "iono_m", "tropo_m", "multipath_excess_m", "cn0_dbhz", "doppler_hz",
        ):
            arr = getattr(ep, name)
            assert arr.shape == (n,), f"{name} shape mismatch"

    def test_to_arrays_shapes_consistent(self, tmp_path):
        nav_file = _write_nav(tmp_path, _NAV_GPS_ONLY)
        result = run_scenario(_base_config(nav_file))
        arrays = result.to_arrays()

        n_rows_expected = sum(ep.n_sat for ep in result.epochs)
        assert n_rows_expected > 0
        for key, arr in arrays.items():
            assert len(arr) == n_rows_expected, f"{key} length mismatch"

        # epoch_index should be monotonic non-decreasing and span [0, n_epochs)
        assert np.all(np.diff(arrays["epoch_index"]) >= 0)
        assert arrays["epoch_index"].max() == result.n_epochs - 1

        # sat_id strings look like "G01"/"G06"
        for sid in np.unique(arrays["sat_id"]):
            assert len(sid) == 3
            assert sid[0].isalpha()

    def test_multi_constellation_sat_ids(self, tmp_path):
        nav_file = _write_nav(tmp_path, _NAV_MIXED, name="mixed.nav")
        result = run_scenario(_base_config(nav_file, constellations=["G", "E"]))
        arrays = result.to_arrays()
        systems = {sid[0] for sid in arrays["sat_id"]}
        assert systems == {"G", "E"}


# ---------------------------------------------------------------------------
# Pseudorange composition
# ---------------------------------------------------------------------------


class TestPseudorangeReconstruction:
    def test_reconstructs_within_tolerance_when_noise_is_zero(self, tmp_path):
        nav_file = _write_nav(tmp_path, _NAV_GPS_ONLY)
        config = _base_config(nav_file, rx_clock_bias_m=1234.5)
        result = run_scenario(config)

        for ep in result.epochs:
            if ep.n_sat == 0:
                continue
            expected = (
                ep.range_geometric_m
                + config.rx_clock_bias_m
                - ep.sat_clock_bias_m
                + ep.iono_m
                + ep.tropo_m
                + ep.multipath_excess_m
            )
            np.testing.assert_allclose(ep.pseudorange_m, expected, atol=1e-6)

    def test_no_plateau_means_zero_multipath_and_all_los(self, tmp_path):
        nav_file = _write_nav(tmp_path, _NAV_GPS_ONLY)
        result = run_scenario(_base_config(nav_file))
        for ep in result.epochs:
            if ep.n_sat == 0:
                continue
            assert np.all(ep.is_los)
            np.testing.assert_array_equal(ep.multipath_excess_m, np.zeros(ep.n_sat))
            np.testing.assert_array_equal(ep.cn0_dbhz >= 0.0, np.ones(ep.n_sat, dtype=bool))

    def test_noise_perturbs_pseudorange(self, tmp_path):
        nav_file = _write_nav(tmp_path, _NAV_GPS_ONLY)
        config = _base_config(
            nav_file,
            pr_noise_sigma_zenith_m=1.0,
            pr_noise_sigma_horizon_m=5.0,
        )
        result = run_scenario(config)
        ep = result.epochs[0]
        expected_noiseless = (
            ep.range_geometric_m
            + config.rx_clock_bias_m
            - ep.sat_clock_bias_m
            + ep.iono_m
            + ep.tropo_m
            + ep.multipath_excess_m
        )
        assert not np.allclose(ep.pseudorange_m, expected_noiseless, atol=1e-6, rtol=0.0)


# ---------------------------------------------------------------------------
# Elevation mask
# ---------------------------------------------------------------------------


class TestElevationMask:
    def test_mask_reduces_visible_count(self, tmp_path):
        nav_file = _write_nav(tmp_path, _NAV_GPS_ONLY)

        low_mask = run_scenario(_base_config(nav_file, elevation_mask_deg=0.0))
        high_mask = run_scenario(_base_config(nav_file, elevation_mask_deg=89.9))

        n_low = sum(ep.n_sat for ep in low_mask.epochs)
        n_high = sum(ep.n_sat for ep in high_mask.epochs)
        assert n_high <= n_low

        for ep in high_mask.epochs:
            if ep.n_sat:
                assert np.all(ep.elevation_rad >= np.radians(89.9))

    def test_mask_can_exclude_all_satellites(self, tmp_path):
        nav_file = _write_nav(tmp_path, _NAV_GPS_ONLY)
        result = run_scenario(_base_config(nav_file, elevation_mask_deg=89.999))
        for ep in result.epochs:
            assert ep.n_sat == 0
            assert ep.sat_id.shape == (0,)
            assert ep.pseudorange_m.shape == (0,)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_reproduces_pseudorange(self, tmp_path):
        nav_file = _write_nav(tmp_path, _NAV_GPS_ONLY)
        config = _base_config(
            nav_file,
            pr_noise_sigma_zenith_m=0.5,
            pr_noise_sigma_horizon_m=2.0,
            seed=7,
        )
        result_a = run_scenario(config)
        result_b = run_scenario(config)

        arrays_a = result_a.to_arrays()
        arrays_b = result_b.to_arrays()
        np.testing.assert_array_equal(arrays_a["pseudorange_m"], arrays_b["pseudorange_m"])

    def test_different_seed_changes_noise(self, tmp_path):
        nav_file = _write_nav(tmp_path, _NAV_GPS_ONLY)
        config_a = _base_config(
            nav_file, pr_noise_sigma_zenith_m=0.5, pr_noise_sigma_horizon_m=2.0, seed=1,
        )
        config_b = _base_config(
            nav_file, pr_noise_sigma_zenith_m=0.5, pr_noise_sigma_horizon_m=2.0, seed=2,
        )
        arrays_a = run_scenario(config_a).to_arrays()
        arrays_b = run_scenario(config_b).to_arrays()
        assert not np.array_equal(arrays_a["pseudorange_m"], arrays_b["pseudorange_m"])


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_requires_receiver_position(self):
        with pytest.raises(ValueError, match="lat/lon/alt or route"):
            ScenarioConfig(nav_file="x.nav", start_time=_START_TIME, duration_s=1.0)

    def test_rejects_both_point_and_route(self):
        with pytest.raises(ValueError, match="not both"):
            ScenarioConfig(
                nav_file="x.nav", lat_deg=1.0, lon_deg=2.0, alt_m=3.0,
                route=[(0.0, 1.0, 2.0, 3.0)],
                start_time=_START_TIME, duration_s=1.0,
            )

    def test_requires_time_window(self):
        with pytest.raises(ValueError, match="start_time"):
            ScenarioConfig(nav_file="x.nav", lat_deg=1.0, lon_deg=2.0, alt_m=3.0)

    def test_rejects_bad_diffraction_model(self):
        with pytest.raises(ValueError, match="diffraction_model"):
            ScenarioConfig(
                nav_file="x.nav", lat_deg=1.0, lon_deg=2.0, alt_m=3.0,
                start_time=_START_TIME, duration_s=1.0, diffraction_model="bogus",
            )

    def test_normalizes_constellation_string(self):
        config = ScenarioConfig(
            nav_file="x.nav", lat_deg=1.0, lon_deg=2.0, alt_m=3.0,
            start_time=_START_TIME, duration_s=1.0, constellations="GEJ",
        )
        assert config.constellations == ("E", "G", "J")


# ---------------------------------------------------------------------------
# Route interpolation
# ---------------------------------------------------------------------------


class TestRoute:
    def test_route_interpolates_receiver_position(self, tmp_path):
        nav_file = _write_nav(tmp_path, _NAV_GPS_ONLY)
        route = [
            ("2024-01-15T00:00:00", 35.0, 139.0, 10.0),
            ("2024-01-15T00:00:02", 35.2, 139.2, 30.0),
        ]
        config = ScenarioConfig(
            nav_file=nav_file,
            route=route,
            start_time=_START_TIME,
            duration_s=2.0,
            step_s=1.0,
            constellations=["G"],
            elevation_mask_deg=0.0,
            pr_noise_sigma_zenith_m=0.0,
            pr_noise_sigma_horizon_m=0.0,
            seed=1,
        )
        result = run_scenario(config)
        assert result.n_epochs == 3

        # rx_ecef should move monotonically between the two waypoints -- check
        # the midpoint epoch sits between the endpoint ECEF positions on each axis.
        rx0 = result.epochs[0].rx_ecef
        rx1 = result.epochs[1].rx_ecef
        rx2 = result.epochs[2].rx_ecef
        for axis in range(3):
            lo, hi = sorted((rx0[axis], rx2[axis]))
            assert lo - 1e-6 <= rx1[axis] <= hi + 1e-6


# ---------------------------------------------------------------------------
# Graceful degradation without a real PLATEAU mesh / GPU
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    def test_los_check_failure_falls_back_to_all_visible(self, tmp_path, monkeypatch):
        nav_file = _write_nav(tmp_path, _NAV_GPS_ONLY)

        class _FakeBuildingModel:
            triangles = np.zeros((0, 3, 3), dtype=np.float64)

            def check_los(self, rx_ecef, sat_ecef):
                raise RuntimeError("no CUDA device available")

        import gnss_gpu.scenario as scenario_mod

        monkeypatch.setattr(
            scenario_mod, "_load_building_model", lambda config, warn_once: _FakeBuildingModel()
        )

        config = _base_config(nav_file, plateau_dir=str(tmp_path))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = run_scenario(config)
        assert any("LOS check unavailable" in str(w.message) for w in caught)
        for ep in result.epochs:
            if ep.n_sat:
                assert np.all(ep.is_los)

    def test_missing_plateau_dir_warns_and_degrades(self, tmp_path):
        nav_file = _write_nav(tmp_path, _NAV_GPS_ONLY)
        missing_dir = tmp_path / "does_not_exist"
        config = _base_config(nav_file, plateau_dir=str(missing_dir))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = run_scenario(config)
        assert any("PLATEAU" in str(w.message) for w in caught)
        for ep in result.epochs:
            if ep.n_sat:
                assert np.all(ep.is_los)
                np.testing.assert_array_equal(ep.multipath_excess_m, np.zeros(ep.n_sat))


# ---------------------------------------------------------------------------
# Optional: real PLATEAU mesh (skipped when the fixture data is not checked out)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_PLATEAU_FIXTURE, reason="experiments/data/plateau_odaiba not available")
class TestRealPlateauMesh:
    def test_run_scenario_with_real_mesh_does_not_crash(self, tmp_path):
        nav_file = _write_nav(tmp_path, _NAV_GPS_ONLY)
        config = _base_config(
            nav_file,
            lat_deg=35.619, lon_deg=139.779, alt_m=30.0,
            plateau_dir=str(_PLATEAU_DIR),
            duration_s=1.0,
        )
        result = run_scenario(config)
        assert result.n_epochs == 2


# ---------------------------------------------------------------------------
# RINEX export
# ---------------------------------------------------------------------------


class TestToRinex:
    def test_round_trips_through_read_rinex_obs(self, tmp_path):
        from gnss_gpu.io.rinex import read_rinex_obs

        nav_file = _write_nav(tmp_path, _NAV_GPS_ONLY)
        result = run_scenario(_base_config(nav_file))
        out_path = tmp_path / "sim.obs"
        result.to_rinex(out_path)

        obs = read_rinex_obs(out_path)
        assert len(obs.epochs) == result.n_epochs

        # No carrier phase was simulated -- L1C must not appear in the header.
        for codes in obs.header.obs_types.values():
            assert "L1C" not in codes
            assert set(codes) == {"C1C", "D1C", "S1C"}

        for expected_ep, actual_ep in zip(result.epochs, obs.epochs):
            assert actual_ep.time == expected_ep.time_utc
            assert list(actual_ep.satellites) == list(expected_ep.sat_id)
            for i, sat_id in enumerate(expected_ep.sat_id):
                sat_obs = actual_ep.observations[sat_id]
                assert sat_obs["C1C"] == pytest.approx(
                    float(expected_ep.pseudorange_m[i]), abs=1e-3
                )
                assert sat_obs["S1C"] == pytest.approx(
                    float(expected_ep.cn0_dbhz[i]), abs=1e-3
                )

    def test_header_approx_position_is_first_epoch_rx_ecef(self, tmp_path):
        nav_file = _write_nav(tmp_path, _NAV_GPS_ONLY)
        result = run_scenario(_base_config(nav_file))
        out_path = tmp_path / "sim2.obs"
        result.to_rinex(out_path)

        text = out_path.read_text()
        pos_line = [ln for ln in text.splitlines() if "APPROX POSITION XYZ" in ln][0]
        x = float(pos_line[0:14])
        assert x == pytest.approx(float(result.epochs[0].rx_ecef[0]), abs=1e-3)

    def test_cli_rinex_out(self, tmp_path):
        import gnss_gpu.scenario_cli as scenario_cli
        from gnss_gpu.io.rinex import read_rinex_obs

        nav_file = _write_nav(tmp_path, _NAV_GPS_ONLY)
        rinex_out = tmp_path / "cli_out.obs"
        argv = [
            "--nav", nav_file,
            "--lat", str(_RX_LAT),
            "--lon", str(_RX_LON),
            "--alt", str(_RX_ALT),
            "--start", _START_TIME,
            "--duration", "2.0",
            "--step", "1.0",
            "--constellations", "G",
            "--elevation-mask-deg", "0.0",
            "--rinex-out", str(rinex_out),
        ]
        rc = scenario_cli.main(argv)
        assert rc == 0
        assert rinex_out.exists()

        obs = read_rinex_obs(rinex_out)
        assert len(obs.epochs) == 3
