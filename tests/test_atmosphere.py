"""Tests for atmospheric delay correction models."""

import numpy as np
import pytest
from gnss_gpu.atmosphere import AtmosphereCorrection, _tropo_saastamoinen_cpu, _iono_klobuchar_cpu


C_LIGHT = 299792458.0


class TestTropoSaastamoinen:
    """Tests for Saastamoinen tropospheric delay model."""

    def test_zenith_delay_sea_level(self):
        """Zenith tropospheric delay should be ~2.3m at sea level."""
        lat = np.radians(45.0)
        alt = 0.0
        el = np.pi / 2.0  # zenith
        delay = _tropo_saastamoinen_cpu(lat, alt, el)
        assert 2.0 < delay < 2.7, f"Zenith delay at sea level: {delay:.3f} m"

    def test_zenith_delay_high_altitude(self):
        """Zenith delay should decrease with altitude."""
        lat = np.radians(45.0)
        el = np.pi / 2.0

        delay_0 = _tropo_saastamoinen_cpu(lat, 0.0, el)
        delay_1000 = _tropo_saastamoinen_cpu(lat, 1000.0, el)
        delay_3000 = _tropo_saastamoinen_cpu(lat, 3000.0, el)

        assert delay_0 > delay_1000 > delay_3000, \
            f"Delays should decrease with altitude: {delay_0:.3f} > {delay_1000:.3f} > {delay_3000:.3f}"

    def test_elevation_dependency(self):
        """Lower elevation should produce larger delay."""
        lat = np.radians(45.0)
        alt = 0.0

        delay_90 = _tropo_saastamoinen_cpu(lat, alt, np.radians(90))  # zenith
        delay_45 = _tropo_saastamoinen_cpu(lat, alt, np.radians(45))
        delay_15 = _tropo_saastamoinen_cpu(lat, alt, np.radians(15))
        delay_5 = _tropo_saastamoinen_cpu(lat, alt, np.radians(5))

        assert delay_5 > delay_15 > delay_45 > delay_90, \
            f"Delays: 5deg={delay_5:.2f}, 15deg={delay_15:.2f}, 45deg={delay_45:.2f}, 90deg={delay_90:.2f}"

    def test_low_elevation_large_delay(self):
        """At 5 degrees elevation, delay should be roughly 10x zenith delay."""
        lat = np.radians(45.0)
        alt = 0.0

        delay_zen = _tropo_saastamoinen_cpu(lat, alt, np.radians(90))
        delay_5 = _tropo_saastamoinen_cpu(lat, alt, np.radians(5))

        ratio = delay_5 / delay_zen
        assert 5 < ratio < 15, f"Ratio 5deg/zenith = {ratio:.1f}, expected ~10"

    def test_vectorized(self):
        """Test with array of elevations."""
        lat = np.radians(35.0)
        alt = 100.0
        elevations = np.radians(np.array([10, 30, 60, 90], dtype=np.float64))

        delays = _tropo_saastamoinen_cpu(lat, alt, elevations)
        assert delays.shape == (4,)
        # Delays should be monotonically decreasing with elevation
        assert all(delays[i] > delays[i+1] for i in range(3))


class TestIonoKlobuchar:
    """Tests for Klobuchar ionospheric delay model."""

    # Typical GPS broadcast parameters
    ALPHA = [1.1176e-8, -7.4506e-9, -5.9605e-8, 1.1921e-7]
    BETA = [1.1264e5, -3.2768e4, -2.6214e5, 4.5875e5]

    def test_nighttime_minimum(self):
        """At nighttime, delay should be ~5ns * C_LIGHT ~= 1.5m (times obliquity)."""
        lat = np.radians(45.0)
        lon = np.radians(0.0)
        az = np.radians(180.0)
        el = np.radians(90.0)  # zenith -> F ~ 1.0
        # Nighttime: local time far from 14:00 (50400s)
        # Use gps_time such that local time is around midnight
        gps_time = 0.0  # midnight at lon=0

        delay = _iono_klobuchar_cpu(self.ALPHA, self.BETA, lat, lon, az, el, gps_time)
        # 5e-9 * C_LIGHT * F (F ~ 1 at zenith)
        expected_min = 5.0e-9 * C_LIGHT
        assert 1.0 < delay < 3.0, f"Nighttime zenith delay: {delay:.3f} m, expected ~{expected_min:.3f}"

    def test_daytime_larger(self):
        """Daytime delay should be larger than nighttime."""
        lat = np.radians(45.0)
        lon = np.radians(0.0)
        az = np.radians(180.0)
        el = np.radians(45.0)

        # Daytime: local time ~14:00 at longitude 0 => gps_time ~50400
        delay_day = _iono_klobuchar_cpu(self.ALPHA, self.BETA, lat, lon, az, el, 50400.0)
        # Nighttime
        delay_night = _iono_klobuchar_cpu(self.ALPHA, self.BETA, lat, lon, az, el, 0.0)

        assert delay_day > delay_night, \
            f"Daytime ({delay_day:.2f}m) should exceed nighttime ({delay_night:.2f}m)"

    def test_obliquity_factor(self):
        """Lower elevation should increase iono delay due to obliquity."""
        lat = np.radians(45.0)
        lon = np.radians(0.0)
        az = np.radians(0.0)
        gps_time = 50400.0

        delay_high = _iono_klobuchar_cpu(self.ALPHA, self.BETA, lat, lon, az, np.radians(70), gps_time)
        delay_low = _iono_klobuchar_cpu(self.ALPHA, self.BETA, lat, lon, az, np.radians(20), gps_time)

        assert delay_low > delay_high, \
            f"Low el ({delay_low:.2f}m) should exceed high el ({delay_high:.2f}m)"

    def test_delay_reasonable_range(self):
        """Ionospheric delay should be in typical range 1-30m."""
        lat = np.radians(35.0)
        lon = np.radians(139.0)
        az = np.radians(45.0)
        el = np.radians(30.0)
        gps_time = 50400.0

        delay = _iono_klobuchar_cpu(self.ALPHA, self.BETA, lat, lon, az, el, gps_time)
        assert 0.5 < delay < 50.0, f"Iono delay = {delay:.2f}m, expected 0.5-50m"

    def test_zero_parameters(self):
        """With zero parameters, delay should be 5ns minimum."""
        alpha_zero = [0.0, 0.0, 0.0, 0.0]
        beta_zero = [0.0, 0.0, 0.0, 0.0]
        lat = np.radians(45.0)
        lon = np.radians(0.0)
        az = np.radians(0.0)
        el = np.radians(45.0)

        delay = _iono_klobuchar_cpu(alpha_zero, beta_zero, lat, lon, az, el, 50400.0)
        F = 1.0 + 16.0 * (0.53 - el / np.pi) ** 3
        expected = 5.0e-9 * C_LIGHT * F
        assert abs(delay - expected) < 0.5, \
            f"Zero-param delay = {delay:.3f}m, expected ~{expected:.3f}m"


class TestAtmosphereCorrection:
    """Tests for the AtmosphereCorrection high-level class."""

    def test_init_defaults(self):
        """Default initialization should set typical iono parameters."""
        atm = AtmosphereCorrection()
        assert len(atm.alpha) == 4
        assert len(atm.beta) == 4

    def test_init_custom(self):
        """Custom iono parameters should be stored."""
        alpha = [1e-8, 2e-8, 3e-8, 4e-8]
        beta = [1e5, 2e5, 3e5, 4e5]
        atm = AtmosphereCorrection(iono_alpha=alpha, iono_beta=beta)
        assert atm.alpha == alpha
        assert atm.beta == beta

    def test_tropo_single_epoch(self):
        """Test tropo correction with single epoch."""
        atm = AtmosphereCorrection()
        rx_lla = np.array([np.radians(45.0), np.radians(0.0), 0.0])
        sat_el = np.radians(np.array([30.0, 60.0, 90.0]))

        delays = atm.tropo(rx_lla, sat_el)
        assert delays.shape == (3,)
        assert all(d > 0 for d in delays)
        # 30 deg should have largest delay
        assert delays[0] > delays[1] > delays[2]

    def test_iono_single_epoch(self):
        """Test iono correction with single epoch."""
        atm = AtmosphereCorrection()
        rx_lla = np.array([np.radians(45.0), np.radians(0.0), 0.0])
        sat_az = np.radians(np.array([0.0, 90.0, 180.0]))
        sat_el = np.radians(np.array([30.0, 45.0, 60.0]))

        delays = atm.iono(rx_lla, sat_az, sat_el, 50400.0)
        assert delays.shape == (3,)
        assert all(d > 0 for d in delays)

    def test_total_is_sum(self):
        """Total correction should equal tropo + iono."""
        atm = AtmosphereCorrection()
        rx_lla = np.array([np.radians(35.0), np.radians(139.0), 50.0])
        sat_az = np.radians(np.array([45.0, 135.0]))
        sat_el = np.radians(np.array([30.0, 60.0]))
        gps_time = 50400.0

        total = atm.total(rx_lla, sat_az, sat_el, gps_time)
        tropo = atm.tropo(rx_lla, sat_el)
        iono = atm.iono(rx_lla, sat_az, sat_el, gps_time)

        np.testing.assert_allclose(total, tropo + iono, rtol=1e-10)

    def test_batch_multi_epoch(self):
        """Test batch processing with multiple epochs."""
        atm = AtmosphereCorrection()
        n_epoch = 3
        n_sat = 4
        rx_lla = np.array([
            [np.radians(35.0), np.radians(139.0), 50.0],
            [np.radians(35.1), np.radians(139.1), 55.0],
            [np.radians(35.2), np.radians(139.2), 60.0],
        ])
        sat_el = np.radians(np.tile([20.0, 40.0, 60.0, 80.0], (n_epoch, 1)))
        sat_az = np.radians(np.tile([0.0, 90.0, 180.0, 270.0], (n_epoch, 1)))
        gps_times = np.array([50000.0, 50200.0, 50400.0])

        tropo = atm.tropo(rx_lla, sat_el)
        assert tropo.shape == (n_epoch, n_sat)

        iono = atm.iono(rx_lla, sat_az, sat_el, gps_times)
        assert iono.shape == (n_epoch, n_sat)

        total = atm.total(rx_lla, sat_az, sat_el, gps_times)
        assert total.shape == (n_epoch, n_sat)
        assert np.all(total > 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
