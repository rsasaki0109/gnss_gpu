"""Tests for SBAS and QZSS augmentation corrections."""

import math

import numpy as np
import pytest

from gnss_gpu.sbas import (
    SBASCorrection,
    QZSSAugmentation,
    IonoGridPoint,
    UDRE_SIGMA,
    GIVE_SIGMA,
    _iono_pierce_point,
    _obliquity_factor,
    _bilinear_interpolate,
)


# ---------------------------------------------------------------------------
# SBAS Fast Correction
# ---------------------------------------------------------------------------

class TestSBASFastCorrection:
    """Tests for SBAS fast clock correction application."""

    def test_apply_correction_basic(self):
        """Pseudorange should be adjusted by PRC."""
        sbas = SBASCorrection()
        sbas.set_fast_correction(prn=3, prc=5.0)

        raw_pr = 20_000_000.0
        corrected = sbas.apply_fast_correction(3, raw_pr)
        assert corrected == pytest.approx(raw_pr + 5.0)

    def test_no_correction_passthrough(self):
        """Unknown PRN should return the original pseudorange."""
        sbas = SBASCorrection()
        raw_pr = 20_000_000.0
        corrected = sbas.apply_fast_correction(99, raw_pr)
        assert corrected == raw_pr

    def test_range_rate_extrapolation(self):
        """Correction should extrapolate using RRC and time difference."""
        sbas = SBASCorrection()
        sbas.set_fast_correction(prn=7, prc=2.0, rrc=0.1, t_apply=100.0)

        corrected = sbas.apply_fast_correction(7, 20_000_000.0, t_current=110.0)
        # correction = 2.0 + 0.1 * (110 - 100) = 3.0
        expected = 20_000_000.0 + 3.0
        assert corrected == pytest.approx(expected)

    def test_negative_correction(self):
        """Negative PRC should reduce pseudorange."""
        sbas = SBASCorrection()
        sbas.set_fast_correction(prn=5, prc=-3.5)

        raw_pr = 20_000_000.0
        corrected = sbas.apply_fast_correction(5, raw_pr)
        assert corrected == pytest.approx(raw_pr - 3.5)

    def test_overwrite_correction(self):
        """Setting a new correction should overwrite the old one."""
        sbas = SBASCorrection()
        sbas.set_fast_correction(prn=3, prc=5.0)
        sbas.set_fast_correction(prn=3, prc=-1.0)

        corrected = sbas.apply_fast_correction(3, 20_000_000.0)
        assert corrected == pytest.approx(20_000_000.0 - 1.0)


# ---------------------------------------------------------------------------
# SBAS Long-Term Correction
# ---------------------------------------------------------------------------

class TestSBASLongTermCorrection:
    """Tests for SBAS long-term orbital/clock corrections."""

    def test_position_correction(self):
        """Satellite position should be offset by dx/dy/dz."""
        sbas = SBASCorrection()
        sbas.set_long_term_correction(prn=10, dx=1.0, dy=-2.0, dz=0.5, da_f0=0.3)

        sat_pos = np.array([10_000_000.0, 20_000_000.0, 30_000_000.0])
        pos_out, clk_out = sbas.apply_long_term_correction(10, sat_pos, 0.0)

        np.testing.assert_allclose(pos_out, sat_pos + [1.0, -2.0, 0.5])
        assert clk_out == pytest.approx(0.3)

    def test_no_correction_passthrough(self):
        """Unknown PRN returns original values."""
        sbas = SBASCorrection()
        sat_pos = np.array([1.0, 2.0, 3.0])
        pos_out, clk_out = sbas.apply_long_term_correction(99, sat_pos, 5.0)

        np.testing.assert_array_equal(pos_out, sat_pos)
        assert clk_out == 5.0


# ---------------------------------------------------------------------------
# SBAS Ionospheric Grid Correction
# ---------------------------------------------------------------------------

class TestSBASIonoGrid:
    """Tests for SBAS ionospheric grid interpolation."""

    @staticmethod
    def _make_grid_2x2():
        """Create a simple 2x2 grid for testing."""
        return [
            IonoGridPoint(lat_deg=30.0, lon_deg=130.0, vertical_delay=2.0, give_index=3),
            IonoGridPoint(lat_deg=30.0, lon_deg=140.0, vertical_delay=4.0, give_index=3),
            IonoGridPoint(lat_deg=40.0, lon_deg=130.0, vertical_delay=6.0, give_index=3),
            IonoGridPoint(lat_deg=40.0, lon_deg=140.0, vertical_delay=8.0, give_index=3),
        ]

    def test_bilinear_center(self):
        """Center of a 2x2 grid should be the average of all 4 values."""
        grid = self._make_grid_2x2()
        result = _bilinear_interpolate(grid, lat_deg=35.0, lon_deg=135.0)
        # t=0.5, u=0.5 => (2*0.25 + 4*0.25 + 6*0.25 + 8*0.25) = 5.0
        assert result == pytest.approx(5.0)

    def test_bilinear_corner(self):
        """Query at a grid point should return that point's value."""
        grid = self._make_grid_2x2()
        result = _bilinear_interpolate(grid, lat_deg=30.0, lon_deg=130.0)
        assert result == pytest.approx(2.0)

    def test_bilinear_edge(self):
        """Query along an edge should linearly interpolate between 2 points."""
        grid = self._make_grid_2x2()
        # Bottom edge (lat=30): lon=135 => midpoint of 2.0 and 4.0 => 3.0
        result = _bilinear_interpolate(grid, lat_deg=30.0, lon_deg=135.0)
        assert result == pytest.approx(3.0)

    def test_bilinear_quarter(self):
        """Query at (32.5, 132.5) should be bilinear with t=0.25, u=0.25."""
        grid = self._make_grid_2x2()
        result = _bilinear_interpolate(grid, lat_deg=32.5, lon_deg=132.5)
        # t=0.25, u=0.25
        expected = (2.0 * 0.75 * 0.75 +
                    4.0 * 0.75 * 0.25 +
                    6.0 * 0.25 * 0.75 +
                    8.0 * 0.25 * 0.25)
        assert result == pytest.approx(expected)

    def test_apply_iono_correction_empty_grid(self):
        """With no grid, iono correction should be zero."""
        sbas = SBASCorrection()
        delay = sbas.apply_iono_correction(
            lat=math.radians(35.0), lon=math.radians(135.0),
            el=math.radians(45.0), az=math.radians(180.0))
        assert delay == 0.0

    def test_apply_iono_correction_with_grid(self):
        """With a loaded grid, iono correction should be positive."""
        sbas = SBASCorrection()
        sbas.set_iono_grid(self._make_grid_2x2())

        delay = sbas.apply_iono_correction(
            lat=math.radians(35.0), lon=math.radians(135.0),
            el=math.radians(45.0), az=math.radians(180.0))
        assert delay > 0.0

    def test_obliquity_increases_at_low_elevation(self):
        """Obliquity factor should be larger at lower elevations."""
        f_high = _obliquity_factor(math.radians(80.0))
        f_low = _obliquity_factor(math.radians(20.0))
        assert f_low > f_high
        assert f_high >= 1.0

    def test_pierce_point_moves_toward_satellite(self):
        """IPP should shift in the direction of the satellite azimuth."""
        lat = math.radians(35.0)
        lon = math.radians(135.0)
        el = math.radians(45.0)

        # Satellite to the north
        ipp_lat_n, _ = _iono_pierce_point(lat, lon, el, az=0.0)
        assert ipp_lat_n > lat

        # Satellite to the south
        ipp_lat_s, _ = _iono_pierce_point(lat, lon, el, az=math.pi)
        assert ipp_lat_s < lat


# ---------------------------------------------------------------------------
# SBAS Integrity
# ---------------------------------------------------------------------------

class TestSBASIntegrity:
    """Tests for SBAS integrity checking."""

    def test_integrity_usable(self):
        """Low UDRE should be usable."""
        sbas = SBASCorrection()
        sbas.set_fast_correction(prn=3, prc=1.0, udre_index=2)

        info = sbas.integrity_check(3)
        assert info["usable"] is True
        assert info["udre_index"] == 2
        assert info["udre_sigma"] == UDRE_SIGMA[2]

    def test_integrity_not_monitored(self):
        """UDRE index 13+ should be not usable."""
        sbas = SBASCorrection()
        sbas.set_fast_correction(prn=3, prc=1.0, udre_index=13)

        info = sbas.integrity_check(3)
        assert info["usable"] is False
        assert info["udre_sigma"] == float("inf")

    def test_integrity_unknown_prn(self):
        """Unknown PRN should be not usable."""
        sbas = SBASCorrection()
        info = sbas.integrity_check(99)
        assert info["usable"] is False
        assert info["udre_sigma"] == float("inf")

    def test_iono_integrity_with_grid(self):
        """Iono integrity should return GIVE for nearest grid point."""
        sbas = SBASCorrection()
        sbas.set_iono_grid([
            IonoGridPoint(lat_deg=35.0, lon_deg=135.0,
                          vertical_delay=3.0, give_index=4),
        ])
        info = sbas.iono_integrity(35.0, 135.0)
        assert info["give_index"] == 4
        assert info["give_sigma"] == GIVE_SIGMA[4]

    def test_iono_integrity_empty_grid(self):
        """Empty grid should report worst-case GIVE."""
        sbas = SBASCorrection()
        info = sbas.iono_integrity(35.0, 135.0)
        assert info["give_index"] == 15
        assert info["give_sigma"] == float("inf")


# ---------------------------------------------------------------------------
# QZSS Augmentation
# ---------------------------------------------------------------------------

class TestQZSSAugmentation:
    """Tests for QZSS PRN identification and CLAS corrections."""

    @pytest.mark.parametrize("prn,expected", [
        (193, True),
        (197, True),
        (202, True),
        (192, False),
        (203, False),
        (1, False),
        (32, False),
    ])
    def test_is_qzss_prn(self, prn, expected):
        """QZSS PRNs should be 193-202."""
        assert QZSSAugmentation.is_qzss_prn(prn) is expected

    def test_clas_code_bias(self):
        """CLAS code bias should be added to pseudorange."""
        qzss = QZSSAugmentation()
        qzss.set_clas_correction(prn=3, code_bias=0.5, phase_bias=0.02,
                                 clock_correction=0.1)

        pr = np.array([20_000_000.0, 21_000_000.0])
        cp = np.array([100_000_000.0, 101_000_000.0])
        prn_list = [3, 7]

        pr_out, cp_out = qzss.apply_clas(pr, cp, prn_list)

        # PRN 3 corrected: code_bias + clock = 0.5 + 0.1 = 0.6
        assert pr_out[0] == pytest.approx(20_000_000.6)
        # PRN 7 not corrected
        assert pr_out[1] == pytest.approx(21_000_000.0)

        # Phase: phase_bias + clock = 0.02 + 0.1 = 0.12
        assert cp_out[0] == pytest.approx(100_000_000.12)
        assert cp_out[1] == pytest.approx(101_000_000.0)

    def test_clas_no_corrections(self):
        """With no CLAS data, measurements should pass through unchanged."""
        qzss = QZSSAugmentation()
        pr = np.array([20_000_000.0])
        cp = np.array([100_000_000.0])

        pr_out, cp_out = qzss.apply_clas(pr, cp, [3])
        np.testing.assert_array_equal(pr_out, pr)
        np.testing.assert_array_equal(cp_out, cp)

    def test_available_corrections(self):
        """available_corrections should list PRNs with CLAS data."""
        qzss = QZSSAugmentation()
        assert qzss.available_corrections() == []

        qzss.set_clas_correction(prn=7, code_bias=0.1)
        qzss.set_clas_correction(prn=3, code_bias=0.2)
        assert qzss.available_corrections() == [3, 7]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
