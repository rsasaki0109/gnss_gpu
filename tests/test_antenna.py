"""Tests for gnss_gpu.antenna (receiver antenna gain patterns)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest

from gnss_gpu.antenna import AntennaPattern
from gnss_gpu.scenario import ScenarioConfig, run_scenario

# ---------------------------------------------------------------------------
# NAV fixture + scenario config helper, mirroring tests/test_scenario.py.
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
# Preset shapes
# ---------------------------------------------------------------------------


class TestPresetShapes:
    def test_isotropic_is_exactly_zero(self):
        pattern = AntennaPattern.preset("isotropic")
        el = np.radians(np.array([0.0, 10.0, 30.0, 60.0, 90.0]))
        az = np.radians(np.array([0.0, 90.0, 180.0, 270.0, 359.0]))
        gain = pattern.gain_db(el, az)
        np.testing.assert_array_equal(gain, np.zeros_like(gain))

    def test_patch_zenith_greater_than_horizon(self):
        pattern = AntennaPattern.preset("patch")
        zenith = pattern.gain_db(np.radians(90.0), 0.0)
        horizon = pattern.gain_db(np.radians(0.0), 0.0)
        assert zenith > horizon
        assert zenith == pytest.approx(3.0, abs=0.05)
        assert horizon == pytest.approx(-5.0, abs=0.05)

    def test_patch_is_azimuth_symmetric(self):
        pattern = AntennaPattern.preset("patch")
        el = np.radians(45.0)
        az_values = np.radians(np.array([0.0, 60.0, 137.0, 200.0, 310.0]))
        gains = pattern.gain_db(np.full(az_values.shape, el), az_values)
        np.testing.assert_allclose(gains, gains[0], atol=1e-9)

    def test_patch_gain_increases_monotonically_with_elevation(self):
        pattern = AntennaPattern.preset("patch")
        el_deg = np.array([0.0, 10.0, 20.0, 30.0, 45.0, 60.0, 75.0, 90.0])
        gains = pattern.gain_db(np.radians(el_deg), np.zeros_like(el_deg))
        assert np.all(np.diff(gains) >= -1e-9)

    def test_helix_zenith_greater_than_horizon(self):
        pattern = AntennaPattern.preset("helix")
        zenith = pattern.gain_db(np.radians(90.0), 0.0)
        horizon = pattern.gain_db(np.radians(0.0), 0.0)
        assert zenith > horizon
        assert zenith == pytest.approx(2.0, abs=0.05)

    def test_smartphone_lossier_than_patch_everywhere(self):
        patch = AntennaPattern.preset("patch")
        phone = AntennaPattern.preset("smartphone")
        el_deg = np.linspace(0.0, 90.0, 37)
        az_deg = np.linspace(0.0, 359.0, 24)
        el_grid, az_grid = np.meshgrid(np.radians(el_deg), np.radians(az_deg), indexing="ij")
        patch_gain = patch.gain_db(el_grid, az_grid)
        phone_gain = phone.gain_db(el_grid, az_grid)
        assert np.all(phone_gain < patch_gain)

    def test_smartphone_has_azimuth_ripple(self):
        pattern = AntennaPattern.preset("smartphone")
        el = np.radians(45.0)
        az_values = np.radians(np.array([0.0, 45.0, 90.0, 135.0, 180.0]))
        gains = pattern.gain_db(np.full(az_values.shape, el), az_values)
        assert np.ptp(gains) > 0.5  # a real (non-degenerate) azimuth ripple

    def test_smartphone_negative_everywhere(self):
        pattern = AntennaPattern.preset("smartphone")
        el_deg = np.linspace(0.0, 90.0, 19)
        az_deg = np.linspace(0.0, 350.0, 16)
        el_grid, az_grid = np.meshgrid(np.radians(el_deg), np.radians(az_deg), indexing="ij")
        gains = pattern.gain_db(el_grid, az_grid)
        assert np.all(gains < 0.0)

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="unknown preset"):
            AntennaPattern.preset("bogus")


# ---------------------------------------------------------------------------
# Interpolation correctness
# ---------------------------------------------------------------------------


class TestInterpolation:
    def _hand_built_pattern(self) -> AntennaPattern:
        # 3 elevation samples x 3 azimuth samples, easy-to-check gain values.
        el_deg = np.array([0.0, 45.0, 90.0])
        az_deg = np.array([0.0, 90.0, 180.0])
        gain_db = np.array(
            [
                [0.0, 10.0, 20.0],
                [100.0, 110.0, 120.0],
                [200.0, 210.0, 220.0],
            ]
        )
        return AntennaPattern.from_table(el_deg, az_deg, gain_db)

    def test_exact_grid_points(self):
        pattern = self._hand_built_pattern()
        assert pattern.gain_db(np.radians(0.0), np.radians(0.0)) == pytest.approx(0.0)
        assert pattern.gain_db(np.radians(45.0), np.radians(90.0)) == pytest.approx(110.0)
        assert pattern.gain_db(np.radians(90.0), np.radians(180.0)) == pytest.approx(220.0)

    def test_midpoint_bilinear_average(self):
        pattern = self._hand_built_pattern()
        # Midpoint of the 4 corners (0,0)=0, (0,90)=10, (45,0)=100, (45,90)=110
        # -> average = 55.0.
        mid = pattern.gain_db(np.radians(22.5), np.radians(45.0))
        assert mid == pytest.approx(55.0, abs=1e-9)

    def test_elevation_clamped_outside_grid(self):
        pattern = self._hand_built_pattern()
        below = pattern.gain_db(np.radians(-30.0), np.radians(0.0))
        above = pattern.gain_db(np.radians(120.0), np.radians(0.0))
        assert below == pytest.approx(0.0)
        assert above == pytest.approx(200.0)

    def test_azimuth_wraps_around_360(self):
        pattern = self._hand_built_pattern()
        # az grid only spans [0, 180]; the constructor should have appended
        # a wrap column back to az=360(=0) equal to the az=0 column, so a
        # query at az=270 interpolates between az=180 (220) and az=360 (200)
        # at elevation=90 -> exact midpoint 210.
        wrapped = pattern.gain_db(np.radians(90.0), np.radians(270.0))
        assert wrapped == pytest.approx(210.0, abs=1e-9)
        # az=350 should be very close to az=0=360's value (200).
        near_zero = pattern.gain_db(np.radians(90.0), np.radians(350.0))
        assert near_zero == pytest.approx(200.0, abs=2.0)


# ---------------------------------------------------------------------------
# Vectorization
# ---------------------------------------------------------------------------


class TestVectorization:
    def test_scalar_input_returns_float(self):
        pattern = AntennaPattern.preset("patch")
        result = pattern.gain_db(np.radians(45.0), np.radians(30.0))
        assert isinstance(result, float)

    def test_1d_array_shape_preserved(self):
        pattern = AntennaPattern.preset("patch")
        el = np.radians(np.linspace(0.0, 90.0, 13))
        az = np.radians(np.linspace(0.0, 350.0, 13))
        result = pattern.gain_db(el, az)
        assert result.shape == (13,)

    def test_2d_array_shape_preserved(self):
        pattern = AntennaPattern.preset("smartphone")
        el = np.radians(np.linspace(0.0, 90.0, 5))
        az = np.radians(np.linspace(0.0, 350.0, 7))
        el_grid, az_grid = np.meshgrid(el, az, indexing="ij")
        result = pattern.gain_db(el_grid, az_grid)
        assert result.shape == (5, 7)

    def test_broadcasting_scalar_azimuth(self):
        pattern = AntennaPattern.preset("patch")
        el = np.radians(np.array([0.0, 30.0, 90.0]))
        result = pattern.gain_db(el, np.radians(0.0))
        assert result.shape == (3,)

    def test_empty_array_input(self):
        pattern = AntennaPattern.preset("patch")
        result = pattern.gain_db(np.zeros(0), np.zeros(0))
        assert result.shape == (0,)


# ---------------------------------------------------------------------------
# from_table validation
# ---------------------------------------------------------------------------


class TestFromTableValidation:
    def test_rejects_non_increasing_elevation(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            AntennaPattern.from_table(
                np.array([0.0, 10.0, 5.0]), np.array([0.0, 180.0]),
                np.zeros((3, 2)),
            )

    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="shape"):
            AntennaPattern.from_table(
                np.array([0.0, 90.0]), np.array([0.0, 180.0]),
                np.zeros((3, 2)),
            )


# ---------------------------------------------------------------------------
# Scenario-level integration
# ---------------------------------------------------------------------------


class TestScenarioIntegration:
    def test_none_and_isotropic_give_identical_cn0(self, tmp_path):
        nav_file = _write_nav(tmp_path, _NAV_GPS_ONLY)
        result_none = run_scenario(_base_config(nav_file, antenna=None))
        result_iso = run_scenario(_base_config(nav_file, antenna="isotropic"))

        arrays_none = result_none.to_arrays()
        arrays_iso = result_iso.to_arrays()
        np.testing.assert_array_equal(arrays_none["cn0_dbhz"], arrays_iso["cn0_dbhz"])

    def test_smartphone_lowers_cn0_everywhere(self, tmp_path):
        nav_file = _write_nav(tmp_path, _NAV_GPS_ONLY)
        result_none = run_scenario(_base_config(nav_file, antenna=None))
        result_phone = run_scenario(_base_config(nav_file, antenna="smartphone"))

        for ep_none, ep_phone in zip(result_none.epochs, result_phone.epochs):
            if ep_none.n_sat == 0:
                continue
            # cn0 is floored at 0.0 -- only assert the strict decrease where
            # the unshaped value was above the floor (otherwise both clamp
            # to 0.0 and equality is expected, not a strict decrease).
            above_floor = ep_none.cn0_dbhz > 0.0
            assert np.all(ep_phone.cn0_dbhz[above_floor] < ep_none.cn0_dbhz[above_floor])
            assert np.all(ep_phone.cn0_dbhz <= ep_none.cn0_dbhz)

    def test_antenna_pattern_instance_accepted_directly(self, tmp_path):
        nav_file = _write_nav(tmp_path, _NAV_GPS_ONLY)
        pattern = AntennaPattern.preset("patch")
        result_by_name = run_scenario(_base_config(nav_file, antenna="patch"))
        result_by_instance = run_scenario(_base_config(nav_file, antenna=pattern))

        arrays_a = result_by_name.to_arrays()
        arrays_b = result_by_instance.to_arrays()
        np.testing.assert_array_equal(arrays_a["cn0_dbhz"], arrays_b["cn0_dbhz"])

    def test_unknown_antenna_name_raises(self, tmp_path):
        nav_file = _write_nav(tmp_path, _NAV_GPS_ONLY)
        with pytest.raises(ValueError, match="unknown preset"):
            run_scenario(_base_config(nav_file, antenna="bogus_antenna"))

    def test_determinism_preserved_with_antenna(self, tmp_path):
        nav_file = _write_nav(tmp_path, _NAV_GPS_ONLY)
        config = _base_config(
            nav_file,
            antenna="smartphone",
            pr_noise_sigma_zenith_m=0.5,
            pr_noise_sigma_horizon_m=2.0,
            seed=7,
        )
        result_a = run_scenario(config)
        result_b = run_scenario(config)

        arrays_a = result_a.to_arrays()
        arrays_b = result_b.to_arrays()
        np.testing.assert_array_equal(arrays_a["cn0_dbhz"], arrays_b["cn0_dbhz"])
        np.testing.assert_array_equal(arrays_a["pseudorange_m"], arrays_b["pseudorange_m"])

    def test_other_fields_unaffected_by_antenna(self, tmp_path):
        nav_file = _write_nav(tmp_path, _NAV_GPS_ONLY)
        result_none = run_scenario(_base_config(nav_file, antenna=None))
        result_phone = run_scenario(_base_config(nav_file, antenna="smartphone"))

        arrays_none = result_none.to_arrays()
        arrays_phone = result_phone.to_arrays()
        for key in ("pseudorange_m", "elevation_rad", "azimuth_rad", "doppler_hz", "is_los"):
            np.testing.assert_array_equal(arrays_none[key], arrays_phone[key])
