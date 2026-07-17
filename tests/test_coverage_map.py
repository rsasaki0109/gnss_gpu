"""Tests for the GPU area-sweep GNSS coverage/accuracy prediction map
(gnss_gpu.coverage_map).

All tests here run without a GPU and without network access unless the
optional experiments/data PLATEAU fixture is present on disk (skipped
otherwise). The NAV fixture is the same small synthetic RINEX 3 fixture
approach used by tests/test_scenario.py (values borrowed from
tests/test_ephemeris.py).
"""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

import numpy as np
import pytest

import gnss_gpu.coverage_map as coverage_map
from gnss_gpu.coverage_map import CoverageMapConfig, CoverageMapResult, run_coverage_map

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PLATEAU_DIR = _REPO_ROOT / "experiments" / "data" / "plateau_odaiba"
_HAS_PLATEAU_FIXTURE = _PLATEAU_DIR.is_dir()

# ---------------------------------------------------------------------------
# Synthetic RINEX 3 NAV fixture (G01/G06 values lifted from
# tests/test_ephemeris.py, same as tests/test_scenario.py).
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

_START_TIME = "2024-01-15T00:00:00"
_RX_LAT = 35.6
_RX_LON = 139.7


def _write_nav(tmp_path: Path, content: str = _NAV_GPS_ONLY, name: str = "test.nav") -> str:
    nav_file = tmp_path / name
    nav_file.write_text(content)
    return str(nav_file)


def _base_config(nav_file: str, **overrides) -> CoverageMapConfig:
    kwargs = dict(
        nav_file=nav_file,
        center_lat_deg=_RX_LAT,
        center_lon_deg=_RX_LON,
        extent_east_m=100.0,
        extent_north_m=50.0,
        cell_size_m=10.0,
        start_time=_START_TIME,
        duration_s=1.0,
        step_s=1.0,
        constellations=["G"],
        plateau_dir=None,
        elevation_mask_deg=0.0,
    )
    kwargs.update(overrides)
    return CoverageMapConfig(**kwargs)


# ---------------------------------------------------------------------------
# Fake ephemeris: fully controlled satellite geometry (known az/el), so
# availability / masking / DOP-through-the-pipeline tests don't depend on
# real Kepler orbit propagation.
# ---------------------------------------------------------------------------


class _FakeEphemeris:
    """Drop-in replacement for gnss_gpu.ephemeris.Ephemeris.

    Satellites sit at a fixed (az_deg, el_deg) as seen from the grid's ENU
    origin, placed at GPS-orbit range (~20,200 km) so within a small grid
    (a few hundred metres) every cell sees essentially the same geometry.
    """

    def __init__(self, az_el_deg, prns=None, range_m=20_200_000.0):
        self._az_el_deg = list(az_el_deg)
        self._prns = list(prns) if prns is not None else [f"G{i+1:02d}" for i in range(len(self._az_el_deg))]
        self._range_m = float(range_m)
        east_vec, north_vec, up_vec = coverage_map._enu_basis(math.radians(_RX_LAT), math.radians(_RX_LON))
        origin_ecef = _lla_to_ecef_for_test(_RX_LAT, _RX_LON, 0.0)
        sats = []
        for az_deg, el_deg in self._az_el_deg:
            az, el = math.radians(az_deg), math.radians(el_deg)
            direction = (
                math.cos(el) * math.sin(az) * east_vec
                + math.cos(el) * math.cos(az) * north_vec
                + math.sin(el) * up_vec
            )
            sats.append(origin_ecef + self._range_m * direction)
        self._sat_ecef = np.array(sats, dtype=np.float64)

    @property
    def available_prns(self):
        return list(self._prns)

    def compute(self, gps_time, prn_list=None, obs_codes=None):
        return self._sat_ecef.copy(), np.zeros(len(self._prns)), list(self._prns)


def _lla_to_ecef_for_test(lat_deg, lon_deg, alt_m):
    from gnss_gpu.scenario import _lla_deg_to_ecef

    return _lla_deg_to_ecef(lat_deg, lon_deg, alt_m)


def _patch_fake_ephemeris(monkeypatch, az_el_deg):
    fake = _FakeEphemeris(az_el_deg)
    monkeypatch.setattr(coverage_map, "read_nav_rinex_multi", lambda *a, **k: {})
    monkeypatch.setattr(coverage_map, "Ephemeris", lambda nav_messages: fake)
    return fake


# ---------------------------------------------------------------------------
# Grid shape / lat-lon correctness
# ---------------------------------------------------------------------------


class TestGrid:
    def test_grid_shape_matches_extent_and_cell_size(self, tmp_path):
        nav_file = _write_nav(tmp_path)
        config = _base_config(nav_file, extent_east_m=100.0, extent_north_m=50.0, cell_size_m=10.0)
        result = run_coverage_map(config)

        assert result.shape == (5, 10)  # n_north=extent_north/cell, n_east=extent_east/cell
        for arr_name in (
            "mean_visible", "mean_los", "los_fraction", "availability",
            "hdop", "vdop", "gdop", "expected_hpe_m", "cell_lat_deg", "cell_lon_deg",
        ):
            assert getattr(result, arr_name).shape == (5, 10), arr_name

    def test_cell_latlon_centered_and_monotonic(self, tmp_path):
        nav_file = _write_nav(tmp_path)
        config = _base_config(nav_file, extent_east_m=100.0, extent_north_m=100.0, cell_size_m=20.0)
        result = run_coverage_map(config)

        # 5x5 grid: lon increases along axis=1 (east), lat increases along axis=0 (north).
        assert np.all(np.diff(result.cell_lon_deg, axis=1) > 0)
        assert np.all(np.diff(result.cell_lat_deg, axis=0) > 0)

        # Grid is centered on (center_lat_deg, center_lon_deg): the mean of the
        # cell centers should closely match the configured center.
        assert result.cell_lat_deg.mean() == pytest.approx(_RX_LAT, abs=1e-6)
        assert result.cell_lon_deg.mean() == pytest.approx(_RX_LON, abs=1e-6)

        # East-West / North-South spacing between adjacent cell centers should
        # be close to cell_size_m (20 m) in real-world distance -- check via
        # an independent WGS-84 meters-per-degree computation (meridian /
        # prime-vertical radius of curvature at the grid latitude).
        a = 6378137.0
        f = 1.0 / 298.257223563
        e2 = f * (2.0 - f)
        lat_rad = math.radians(_RX_LAT)
        meridian_radius = a * (1.0 - e2) / (1.0 - e2 * math.sin(lat_rad) ** 2) ** 1.5
        prime_vertical_radius = a / math.sqrt(1.0 - e2 * math.sin(lat_rad) ** 2)
        meters_per_deg_lat = meridian_radius * math.pi / 180.0
        meters_per_deg_lon = prime_vertical_radius * math.cos(lat_rad) * math.pi / 180.0

        dlon = result.cell_lon_deg[0, 1] - result.cell_lon_deg[0, 0]
        dlat = result.cell_lat_deg[1, 0] - result.cell_lat_deg[0, 0]
        assert dlon * meters_per_deg_lon == pytest.approx(20.0, rel=1e-4)
        assert dlat * meters_per_deg_lat == pytest.approx(20.0, rel=1e-4)

    def test_extent_smaller_than_one_cell_still_yields_one_cell(self, tmp_path):
        nav_file = _write_nav(tmp_path)
        config = _base_config(nav_file, extent_east_m=3.0, extent_north_m=3.0, cell_size_m=10.0)
        result = run_coverage_map(config)
        assert result.shape == (1, 1)


# ---------------------------------------------------------------------------
# DOP: independent hand-computed 4-satellite geometry
# ---------------------------------------------------------------------------


class TestDOP:
    def test_dop_matches_hand_computed_four_satellite_geometry(self):
        def unit_enu(az_deg, el_deg):
            az, el = math.radians(az_deg), math.radians(el_deg)
            return [math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el)]

        sats = [unit_enu(0.0, 60.0), unit_enu(90.0, 50.0), unit_enu(180.0, 70.0), unit_enu(270.0, 40.0)]
        unit_arr = np.array([sats], dtype=np.float64)  # (1 cell, 4 sats, 3)
        mask = np.ones((1, 4), dtype=bool)

        hdop, vdop, gdop = coverage_map._dop_from_mask(unit_arr, mask)

        # Independent direct computation: build H and (H^T H)^-1 by hand.
        H = np.array([[e, n, u, 1.0] for e, n, u in sats], dtype=np.float64)
        Q = np.linalg.inv(H.T @ H)
        expected_hdop = math.sqrt(Q[0, 0] + Q[1, 1])
        expected_vdop = math.sqrt(Q[2, 2])
        expected_gdop = math.sqrt(np.trace(Q))

        assert hdop[0] == pytest.approx(expected_hdop, abs=1e-6)
        assert vdop[0] == pytest.approx(expected_vdop, abs=1e-6)
        assert gdop[0] == pytest.approx(expected_gdop, abs=1e-6)

    def test_dop_is_nan_below_four_satellites(self):
        def unit_enu(az_deg, el_deg):
            az, el = math.radians(az_deg), math.radians(el_deg)
            return [math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el)]

        sats = [unit_enu(0.0, 60.0), unit_enu(120.0, 50.0), unit_enu(240.0, 40.0)]
        unit_arr = np.array([sats], dtype=np.float64)
        mask = np.ones((1, 3), dtype=bool)

        hdop, vdop, gdop = coverage_map._dop_from_mask(unit_arr, mask)
        assert np.isnan(hdop[0])
        assert np.isnan(vdop[0])
        assert np.isnan(gdop[0])

    def test_dop_masks_out_nlos_satellites(self):
        def unit_enu(az_deg, el_deg):
            az, el = math.radians(az_deg), math.radians(el_deg)
            return [math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el)]

        sats = [unit_enu(0.0, 60.0), unit_enu(90.0, 50.0), unit_enu(180.0, 70.0), unit_enu(270.0, 40.0)]
        unit_arr = np.array([sats], dtype=np.float64)
        # Mask out the 4th satellite -> only 3 counted -> NaN.
        mask = np.array([[True, True, True, False]])

        hdop, _, _ = coverage_map._dop_from_mask(unit_arr, mask)
        assert np.isnan(hdop[0])


# ---------------------------------------------------------------------------
# Availability / elevation-mask logic (fully controlled satellite geometry)
# ---------------------------------------------------------------------------


class TestAvailability:
    def test_availability_true_when_four_sats_above_mask(self, tmp_path, monkeypatch):
        nav_file = _write_nav(tmp_path)
        # 5 satellites at el = 80, 70, 60, 15, 5 degrees, spread in azimuth.
        az_el = [(0.0, 80.0), (72.0, 70.0), (144.0, 60.0), (216.0, 15.0), (288.0, 5.0)]
        _patch_fake_ephemeris(monkeypatch, az_el)

        config = _base_config(
            nav_file, extent_east_m=30.0, extent_north_m=30.0, cell_size_m=10.0,
            elevation_mask_deg=10.0,
        )
        result = run_coverage_map(config)

        # 4 satellites (80,70,60,15) are above a 10 deg mask -> availability=1 everywhere.
        assert np.all(result.mean_visible == 4.0)
        assert np.all(result.availability == 1.0)
        assert np.all(np.isfinite(result.hdop))

    def test_availability_false_when_mask_excludes_below_four(self, tmp_path, monkeypatch):
        nav_file = _write_nav(tmp_path)
        az_el = [(0.0, 80.0), (72.0, 70.0), (144.0, 60.0), (216.0, 15.0), (288.0, 5.0)]
        _patch_fake_ephemeris(monkeypatch, az_el)

        config = _base_config(
            nav_file, extent_east_m=30.0, extent_north_m=30.0, cell_size_m=10.0,
            elevation_mask_deg=30.0,
        )
        result = run_coverage_map(config)

        # Only 3 satellites (80,70,60) survive a 30 deg mask -> never >=4.
        assert np.all(result.mean_visible == 3.0)
        assert np.all(result.availability == 0.0)
        assert np.all(np.isnan(result.hdop))

    def test_mask_reduces_visible_count(self, tmp_path):
        nav_file = _write_nav(tmp_path)
        low_mask = run_coverage_map(_base_config(nav_file, elevation_mask_deg=0.0))
        high_mask = run_coverage_map(_base_config(nav_file, elevation_mask_deg=89.9))
        assert np.all(high_mask.mean_visible <= low_mask.mean_visible)


# ---------------------------------------------------------------------------
# Open-sky (no mesh) path
# ---------------------------------------------------------------------------


class TestOpenSky:
    def test_no_mesh_gives_finite_metrics_everywhere(self, tmp_path, monkeypatch):
        nav_file = _write_nav(tmp_path)
        az_el = [(0.0, 60.0), (72.0, 55.0), (144.0, 65.0), (216.0, 50.0), (288.0, 45.0)]
        _patch_fake_ephemeris(monkeypatch, az_el)

        config = _base_config(
            nav_file, extent_east_m=30.0, extent_north_m=30.0, cell_size_m=10.0,
            elevation_mask_deg=10.0, plateau_dir=None,
        )
        result = run_coverage_map(config)

        for arr_name in ("mean_visible", "mean_los", "los_fraction", "availability", "hdop", "vdop", "gdop", "expected_hpe_m"):
            arr = getattr(result, arr_name)
            assert np.all(np.isfinite(arr)), f"{arr_name} has non-finite entries in open sky"

        assert np.all(result.los_fraction == 1.0)
        assert np.all(result.mean_visible == result.mean_los)
        assert np.all(result.expected_hpe_m == pytest.approx(result.hdop * config.uere_m))

    def test_no_plateau_dir_means_no_building_mask(self, tmp_path):
        nav_file = _write_nav(tmp_path)
        result = run_coverage_map(_base_config(nav_file, plateau_dir=None))
        assert not np.any(np.isnan(result.mean_visible))


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_config_reproduces_all_metrics(self, tmp_path, monkeypatch):
        nav_file = _write_nav(tmp_path)
        az_el = [(0.0, 60.0), (72.0, 55.0), (144.0, 65.0), (216.0, 50.0), (288.0, 45.0)]
        _patch_fake_ephemeris(monkeypatch, az_el)

        config = _base_config(
            nav_file, extent_east_m=30.0, extent_north_m=30.0, cell_size_m=10.0,
            elevation_mask_deg=10.0,
        )
        result_a = run_coverage_map(config)
        result_b = run_coverage_map(config)

        for name in ("mean_visible", "mean_los", "los_fraction", "availability", "hdop", "vdop", "gdop", "expected_hpe_m", "cell_lat_deg", "cell_lon_deg"):
            np.testing.assert_array_equal(getattr(result_a, name), getattr(result_b, name))


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_requires_time_window(self):
        with pytest.raises(ValueError, match="start_time"):
            CoverageMapConfig(nav_file="x.nav", center_lat_deg=1.0, center_lon_deg=2.0)

    def test_rejects_nonpositive_extent(self):
        with pytest.raises(ValueError, match="extent_east_m"):
            CoverageMapConfig(
                nav_file="x.nav", center_lat_deg=1.0, center_lon_deg=2.0,
                extent_east_m=0.0, start_time=_START_TIME, duration_s=1.0,
            )

    def test_rejects_nonpositive_cell_size(self):
        with pytest.raises(ValueError, match="cell_size_m"):
            CoverageMapConfig(
                nav_file="x.nav", center_lat_deg=1.0, center_lon_deg=2.0,
                cell_size_m=0.0, start_time=_START_TIME, duration_s=1.0,
            )


# ---------------------------------------------------------------------------
# Optional: real PLATEAU mesh (skipped when the fixture data is not checked out)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_PLATEAU_FIXTURE, reason="experiments/data/plateau_odaiba not available")
class TestRealPlateauMesh:
    def test_run_coverage_map_with_real_mesh_does_not_crash(self, tmp_path):
        nav_file = _write_nav(tmp_path)
        config = _base_config(
            nav_file,
            center_lat_deg=35.619, center_lon_deg=139.779,
            extent_east_m=60.0, extent_north_m=60.0, cell_size_m=15.0,
            plateau_dir=str(_PLATEAU_DIR),
        )
        result = run_coverage_map(config)
        assert result.shape == (4, 4)
        # Some cells may be NaN (inside a building footprint); shapes must
        # still be consistent and lat/lon must always be finite.
        assert np.all(np.isfinite(result.cell_lat_deg))
        assert np.all(np.isfinite(result.cell_lon_deg))
