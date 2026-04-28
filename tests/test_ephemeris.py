"""Tests for broadcast ephemeris computation."""

import io
import math
import textwrap
from datetime import datetime

import numpy as np
import pytest

from gnss_gpu.io.nav_rinex import NavMessage, read_gps_klobuchar_from_nav_header, read_nav_rinex, _parse_nav_float
from gnss_gpu.ephemeris import (
    Ephemeris,
    GPS_MU,
    GPS_OMEGA_E,
    GPS_F,
    GPS_WEEK_SEC,
    _normalize_sat_id,
)

try:
    from gnss_gpu._gnss_gpu_ephemeris import compute_satellite_position
    HAS_GPU = True
except ImportError:
    HAS_GPU = False


# ---------------------------------------------------------------------------
# Reference ephemeris data for PRN 01 (realistic values from a GPS almanac)
# These values are representative of GPS Block IIF satellite parameters.
# ---------------------------------------------------------------------------
def _make_reference_nav() -> NavMessage:
    """Create a reference NavMessage with known orbital parameters."""
    return NavMessage(
        prn=1,
        toc=datetime(2024, 1, 15, 0, 0, 0),
        af0=-3.930553793907e-05,
        af1=-1.023181539495e-12,
        af2=0.0,
        sqrt_a=5153.637939453125,
        e=0.005765914916992,
        i0=0.973496578994,
        omega0=-0.249523028508,
        omega=0.685940414073,
        M0=1.245932843990,
        delta_n=4.623016997497e-09,
        omega_dot=-8.120689826012e-09,
        idot=1.132502065007e-10,
        cuc=-3.464519977570e-06,
        cus=7.525086402893e-06,
        crc=224.15625,
        crs=-14.03125,
        cic=-1.303851604462e-07,
        cis=5.587935447693e-08,
        toe=518400.0,  # Monday 00:00:00 GPS time
        week=2295,
        tgd=-1.117587089539e-08,
        toc_seconds=518400.0,
    )


# ---------------------------------------------------------------------------
# Test: Kepler equation solver
# ---------------------------------------------------------------------------
class TestKeplerEquation:
    def test_circular_orbit(self):
        """For e=0, E should equal M."""
        E = Ephemeris._kepler_cpu(1.0, 0.0)
        assert abs(E - 1.0) < 1e-14

    def test_low_eccentricity(self):
        """Check convergence for typical GPS eccentricity."""
        M = 1.5
        e = 0.01
        E = Ephemeris._kepler_cpu(M, e)
        # Verify: M = E - e*sin(E)
        residual = abs(M - (E - e * math.sin(E)))
        assert residual < 1e-14

    def test_moderate_eccentricity(self):
        """Check convergence for moderate eccentricity."""
        M = 2.0
        e = 0.3
        E = Ephemeris._kepler_cpu(M, e)
        residual = abs(M - (E - e * math.sin(E)))
        assert residual < 1e-14

    def test_high_eccentricity(self):
        """Check convergence for high eccentricity (beyond GPS, but tests robustness)."""
        M = 0.5
        e = 0.9
        E = Ephemeris._kepler_cpu(M, e)
        residual = abs(M - (E - e * math.sin(E)))
        assert residual < 1e-12

    def test_zero_mean_anomaly(self):
        """E=0 when M=0 for any eccentricity."""
        for e in [0.0, 0.01, 0.5]:
            E = Ephemeris._kepler_cpu(0.0, e)
            assert abs(E) < 1e-14

    def test_pi_mean_anomaly(self):
        """E=pi when M=pi for any eccentricity."""
        for e in [0.0, 0.01, 0.5]:
            E = Ephemeris._kepler_cpu(math.pi, e)
            assert abs(E - math.pi) < 1e-13


# ---------------------------------------------------------------------------
# Test: Satellite position computation (CPU fallback)
# ---------------------------------------------------------------------------
class TestSatellitePosition:
    def test_position_altitude(self):
        """Computed satellite position should be at ~26,560 km altitude (GPS orbit)."""
        nav = _make_reference_nav()
        eph = Ephemeris({1: [nav]})
        pos, clk, prns = eph.compute(nav.toe)

        assert len(prns) == 1
        assert prns[0] == 1

        # Distance from Earth center
        r = np.linalg.norm(pos[0])
        # GPS orbit radius is ~26,560 km (semi-major axis ~26,560 km)
        expected_a = nav.sqrt_a ** 2
        assert abs(r - expected_a) < 100e3  # within 100 km (due to eccentricity)
        assert r > 20000e3  # definitely above LEO
        assert r < 30000e3  # definitely not beyond GEO

    def test_position_deterministic(self):
        """Same inputs should produce same outputs."""
        nav = _make_reference_nav()
        eph = Ephemeris({1: [nav]})
        pos1, clk1, _ = eph.compute(nav.toe)
        pos2, clk2, _ = eph.compute(nav.toe)
        np.testing.assert_array_equal(pos1, pos2)
        np.testing.assert_array_equal(clk1, clk2)

    def test_compute_single_can_omit_group_delay_for_rtklib_satposs_parity(self):
        """MatRTKLIB satposs exposes satellite clock without TGD/BGD code bias."""
        nav = _make_reference_nav()

        _pos_with_delay, clk_with_delay = Ephemeris._compute_single_cpu(nav, nav.toe, "C1")
        _pos_without_delay, clk_without_delay = Ephemeris._compute_single_cpu(
            nav,
            nav.toe,
            "C1",
            apply_group_delay=False,
        )

        assert np.isclose(clk_without_delay - clk_with_delay, nav.tgd)

    def test_position_changes_with_time(self):
        """Satellite position should change over time."""
        nav = _make_reference_nav()
        eph = Ephemeris({1: [nav]})
        pos1, _, _ = eph.compute(nav.toe)
        pos2, _, _ = eph.compute(nav.toe + 300.0)  # 5 minutes later
        diff = np.linalg.norm(pos1[0] - pos2[0])
        # GPS satellite moves ~3.9 km/s, in 300s ~ 1170 km
        assert diff > 100e3  # at least 100 km
        assert diff < 5000e3  # not more than 5000 km in 5 min

    def test_multiple_satellites(self):
        """Test computation with multiple PRNs."""
        nav1 = _make_reference_nav()
        nav2 = NavMessage(
            prn=6,
            toc=datetime(2024, 1, 15, 0, 0, 0),
            af0=1.5e-05,
            af1=0.0,
            af2=0.0,
            sqrt_a=5153.5,
            e=0.008,
            i0=0.97,
            omega0=1.2,
            omega=-0.5,
            M0=2.0,
            delta_n=4.5e-09,
            omega_dot=-8.0e-09,
            idot=1.0e-10,
            cuc=-3.0e-06,
            cus=7.0e-06,
            crc=220.0,
            crs=-10.0,
            cic=-1.0e-07,
            cis=5.0e-08,
            toe=518400.0,
            week=2295,
            tgd=0.0,
            toc_seconds=518400.0,
        )
        eph = Ephemeris({1: [nav1], 6: [nav2]})
        pos, clk, prns = eph.compute(518400.0)
        assert len(prns) == 2
        assert pos.shape == (2, 3)
        assert clk.shape == (2,)
        # Different satellites should have different positions
        assert np.linalg.norm(pos[0] - pos[1]) > 1000.0

    def test_ecef_sum_check(self):
        """ECEF position should satisfy x^2 + y^2 + z^2 ~ a^2 (orbital radius)."""
        nav = _make_reference_nav()
        eph = Ephemeris({1: [nav]})
        pos, _, _ = eph.compute(nav.toe + 1000.0)
        r = np.linalg.norm(pos[0])
        a = nav.sqrt_a ** 2
        # r should be within e*a of a
        assert abs(r - a) < nav.e * a * 1.5  # generous tolerance


# ---------------------------------------------------------------------------
# Test: Clock correction
# ---------------------------------------------------------------------------
class TestClockCorrection:
    def test_clock_at_toc(self):
        """At toc, clock correction should be approximately af0 + relativistic - tgd."""
        nav = _make_reference_nav()
        eph = Ephemeris({1: [nav]})
        _, clk, _ = eph.compute(nav.toe)
        # At toe=toc, dt=0, so clk ~ af0 + dtr - tgd
        # Relativistic correction is small, so clk should be close to af0 - tgd
        expected_approx = nav.af0 - nav.tgd
        # Allow some margin for relativistic correction (a few ns)
        assert abs(clk[0] - expected_approx) < 1e-6

    def test_clock_drift(self):
        """Clock correction should drift with af1 over time."""
        nav = _make_reference_nav()
        nav.af0 = 1e-6  # 1 microsecond bias
        nav.af1 = 1e-10  # 0.1 ns/s drift
        nav.af2 = 0.0
        nav.tgd = 0.0
        eph = Ephemeris({1: [nav]})

        _, clk0, _ = eph.compute(nav.toe)
        _, clk1, _ = eph.compute(nav.toe + 3600.0)  # 1 hour later

        # Expected drift: af1 * 3600 = 3.6e-7
        drift = clk1[0] - clk0[0]
        expected_drift = nav.af1 * 3600.0
        # Allow for relativistic correction difference
        assert abs(drift - expected_drift) < 1e-8

    def test_galileo_e1_clock_uses_bgd_e5b_e1(self):
        """Galileo E1 single-frequency clock should use the E5b/E1 BGD."""
        nav = _make_reference_nav()
        nav.prn = 1
        nav.system = "E"
        nav.data_sources = 513.0
        nav.codes_on_l2 = 513.0
        nav.bgd_e5a_e1 = 4.19095158577e-09
        nav.bgd_e5b_e1 = 8.38190317154e-09
        nav.tgd = nav.bgd_e5a_e1
        nav.e = 0.0
        nav.af1 = 0.0
        nav.af2 = 0.0
        eph = Ephemeris({"E01": [nav]})

        _, clk, _ = eph.compute(nav.toe, ["E01"], obs_codes=["C1C"])
        assert abs(clk[0] - (nav.af0 - nav.bgd_e5b_e1)) < 1e-12


# ---------------------------------------------------------------------------
# Test: NAV RINEX parsing
# ---------------------------------------------------------------------------
class TestNavRinexParsing:
    def test_parse_nav_float(self):
        """Test RINEX float parser with D exponent notation."""
        assert abs(_parse_nav_float("0.100000000000D+01") - 1.0) < 1e-15
        assert abs(_parse_nav_float("-3.930553793907D-05") - -3.930553793907e-05) < 1e-20
        assert abs(_parse_nav_float("5.153637939453D+03") - 5153.637939453) < 1e-9
        assert _parse_nav_float("") == 0.0
        assert abs(_parse_nav_float("1.23E+02") - 123.0) < 1e-10

    def test_read_gps_klobuchar_from_rinex2_header(self, tmp_path):
        nav_content = textwrap.dedent("""\
             2.11           N: GPS NAV DATA                         RINEX VERSION / TYPE
                4.6566D-09  1.4901D-08 -5.9605D-08 -5.9605D-08          ION ALPHA
                7.7824D+04  4.9152D+04 -6.5536D+04 -3.2768D+05          ION BETA
                                                                    END OF HEADER
        """)
        nav_file = tmp_path / "rinex2.nav"
        nav_file.write_text(nav_content)

        alpha, beta = read_gps_klobuchar_from_nav_header(nav_file)

        np.testing.assert_allclose(alpha, [4.6566e-09, 1.4901e-08, -5.9605e-08, -5.9605e-08])
        np.testing.assert_allclose(beta, [7.7824e04, 4.9152e04, -6.5536e04, -3.2768e05])

    def test_parse_v3_nav_file(self, tmp_path):
        """Test parsing a RINEX 3 navigation file."""
        nav_content = textwrap.dedent("""\
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
        """)

        nav_file = tmp_path / "test.nav"
        nav_file.write_text(nav_content)

        result = read_nav_rinex(str(nav_file))

        assert 1 in result
        assert len(result[1]) == 1
        nav = result[1][0]
        assert nav.prn == 1
        assert nav.toc == datetime(2024, 1, 15, 0, 0, 0)
        assert abs(nav.af0 - (-3.930553793907e-05)) < 1e-17
        assert abs(nav.sqrt_a - 5153.637939453) < 1e-6
        assert abs(nav.e - 5.765914916992e-03) < 1e-14
        assert abs(nav.toe - 518400.0) < 1e-6
        assert nav.week == 2295

    def test_parse_multiple_satellites(self, tmp_path):
        """Test parsing navigation file with multiple satellites."""
        nav_content = textwrap.dedent("""\
             3.04           N: GNSS NAV DATA    G: GPS              RINEX VERSION / TYPE
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

        nav_file = tmp_path / "multi.nav"
        nav_file.write_text(nav_content)

        result = read_nav_rinex(str(nav_file))
        assert 1 in result
        assert 6 in result
        assert result[6][0].prn == 6
        assert abs(result[6][0].e - 0.008) < 1e-10

    def test_parse_galileo_data_sources_and_bgd(self, tmp_path):
        """Galileo records should preserve data-source and both BGD fields."""
        nav_content = textwrap.dedent("""\
             3.04           N: GNSS NAV DATA    M: MIXED            RINEX VERSION / TYPE
                                                                    END OF HEADER
            E01 2024 01 15 00 00 00 1.000000000000E-06 0.000000000000E+00 0.000000000000E+00
                 1.000000000000E+00 2.000000000000E+00 3.000000000000E+00 4.000000000000E+00
                 5.000000000000E+00 6.000000000000E+00 7.000000000000E+00 8.000000000000E+00
                 9.000000000000E+00 1.000000000000E+01 1.100000000000E+01 1.200000000000E+01
                 1.300000000000E+01 1.400000000000E+01 1.500000000000E+01 1.600000000000E+01
                 1.700000000000E+01 5.130000000000E+02 2.324000000000E+03 0.000000000000E+00
                 3.120000000000E+01 0.000000000000E+00 4.190951585770E-09 4.423782229420E-09
                 1.958950000000E+05 0.000000000000E+00 0.000000000000E+00 0.000000000000E+00
        """)

        nav_file = tmp_path / "gal.nav"
        nav_file.write_text(nav_content)

        result = read_nav_rinex(str(nav_file), systems=("E",), key_by_sat_id=True)
        nav = result["E01"][0]
        assert nav.system == "E"
        assert nav.data_sources == pytest.approx(513.0)
        assert nav.bgd_e5a_e1 == pytest.approx(4.19095158577e-09)
        assert nav.bgd_e5b_e1 == pytest.approx(4.42378222942e-09)
        assert nav.tgd == pytest.approx(nav.bgd_e5a_e1)


# ---------------------------------------------------------------------------
# Test: Ephemeris selection
# ---------------------------------------------------------------------------
class TestEphemerisSelection:
    def test_normalize_sat_id_with_internal_spaces(self):
        assert _normalize_sat_id("G 5") == "G05"
        assert _normalize_sat_id("E 1") == "E01"
        assert _normalize_sat_id("J02") == "J02"

    def test_select_closest_toe(self):
        """Should select ephemeris with toe closest to requested time."""
        nav1 = _make_reference_nav()
        nav1.toe = 0.0
        nav1.toc_seconds = 0.0

        nav2 = _make_reference_nav()
        nav2.toe = 7200.0  # 2 hours later
        nav2.toc_seconds = 7200.0

        eph = Ephemeris({1: [nav1, nav2]})

        # At t=1000, nav1 (toe=0) is closer
        selected = eph.select_ephemeris(1, 1000.0)
        assert selected.toe == 0.0

        # At t=5000, nav2 (toe=7200) is closer
        selected = eph.select_ephemeris(1, 5000.0)
        assert selected.toe == 7200.0

    def test_select_nonexistent_prn(self):
        """Should return None for unknown PRN."""
        eph = Ephemeris({1: [_make_reference_nav()]})
        assert eph.select_ephemeris(99, 518400.0) is None

    def test_select_galileo_e1_prefers_inav_message(self):
        """E1 observations should prefer Galileo I/NAV records over F/NAV."""
        base = _make_reference_nav()
        base.prn = 1
        base.system = "E"
        base.toe = 1000.0
        base.toc_seconds = 1000.0

        fnav = NavMessage(**{**base.__dict__, "data_sources": 258.0, "codes_on_l2": 258.0})
        e5b_inav = NavMessage(**{**base.__dict__, "data_sources": 516.0, "codes_on_l2": 516.0})
        e1_inav = NavMessage(**{**base.__dict__, "data_sources": 513.0, "codes_on_l2": 513.0})

        eph = Ephemeris({"E01": [fnav, e5b_inav, e1_inav]})
        selected = eph.select_ephemeris("E01", 1000.0, obs_code="C1C")
        assert selected is not None
        assert int(selected.data_sources) == 513


# ---------------------------------------------------------------------------
# Test: Batch computation
# ---------------------------------------------------------------------------
class TestBatchComputation:
    def test_batch_matches_single(self):
        """Batch computation should match individual computations."""
        nav = _make_reference_nav()
        eph = Ephemeris({1: [nav]})

        times = [nav.toe, nav.toe + 600.0, nav.toe + 1200.0]

        # Individual computations
        single_positions = []
        single_clocks = []
        for t in times:
            pos, clk, _ = eph.compute(t)
            single_positions.append(pos[0])
            single_clocks.append(clk[0])

        # Batch computation
        batch_pos, batch_clk, _ = eph.compute_batch(times)

        for i in range(3):
            np.testing.assert_allclose(batch_pos[i, 0], single_positions[i], atol=1e-6)
            np.testing.assert_allclose(batch_clk[i, 0], single_clocks[i], atol=1e-15)

    def test_batch_reselects_ephemeris_across_epochs(self):
        """Batch path should re-select the closest record for each epoch."""
        nav1 = _make_reference_nav()
        nav1.toe = 0.0
        nav1.toc_seconds = 0.0

        nav2 = NavMessage(**{
            **nav1.__dict__,
            "toe": 7200.0,
            "toc_seconds": 7200.0,
            "M0": nav1.M0 + 0.4,
        })

        eph = Ephemeris({1: [nav1, nav2]})
        times = np.array([1000.0, 7400.0])

        batch_pos, batch_clk, prns = eph.compute_batch(times)
        single_pos = []
        single_clk = []
        for t in times:
            pos, clk, _ = eph.compute(float(t))
            single_pos.append(pos[0])
            single_clk.append(clk[0])

        assert prns == [1]
        np.testing.assert_allclose(batch_pos[:, 0], np.array(single_pos), atol=1e-6)
        np.testing.assert_allclose(batch_clk[:, 0], np.array(single_clk), atol=1e-15)

    def test_batch_shape(self):
        """Batch output should have correct shape."""
        nav1 = _make_reference_nav()
        nav2 = NavMessage(
            prn=6, toc=datetime(2024, 1, 15, 0, 0, 0),
            af0=0.0, af1=0.0, af2=0.0,
            sqrt_a=5153.5, e=0.008, i0=0.97, omega0=1.2, omega=-0.5,
            M0=2.0, delta_n=4.5e-09, omega_dot=-8.0e-09, idot=1.0e-10,
            cuc=-3.0e-06, cus=7.0e-06, crc=220.0, crs=-10.0,
            cic=-1.0e-07, cis=5.0e-08, toe=518400.0, week=2295,
            tgd=0.0, toc_seconds=518400.0,
        )
        eph = Ephemeris({1: [nav1], 6: [nav2]})

        times = np.array([518400.0, 518700.0, 519000.0])
        pos, clk, prns = eph.compute_batch(times)

        assert pos.shape == (3, 2, 3)
        assert clk.shape == (3, 2)
        assert len(prns) == 2


# ---------------------------------------------------------------------------
# Test: GPU computation (skip if no GPU)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not HAS_GPU, reason="CUDA module not available")
class TestGPUComputation:
    def test_gpu_matches_cpu(self):
        """GPU results should match CPU fallback within floating-point tolerance."""
        nav = _make_reference_nav()
        eph = Ephemeris({1: [nav]})

        # Force CPU computation
        cpu_pos, cpu_clk, _ = eph._compute_cpu(nav.toe + 600.0)

        # GPU computation
        gpu_pos, gpu_clk, _ = eph.compute(nav.toe + 600.0)

        np.testing.assert_allclose(gpu_pos, cpu_pos, atol=1e-3)
        np.testing.assert_allclose(gpu_clk, cpu_clk, atol=1e-12)

    def test_gpu_batch(self):
        """GPU batch computation should produce reasonable results."""
        nav = _make_reference_nav()
        eph = Ephemeris({1: [nav]})

        times = np.array([nav.toe + i * 30.0 for i in range(100)])
        pos, clk, prns = eph.compute_batch(times)

        assert pos.shape == (100, 1, 3)
        assert clk.shape == (100, 1)
        # All positions should be at GPS orbital altitude
        for i in range(100):
            r = np.linalg.norm(pos[i, 0])
            assert 25000e3 < r < 28000e3
