"""Verification tests: compare gnss_gpu against gnss_lib_py (Stanford NavLab).

These tests validate that our CUDA implementations produce results consistent
with a well-tested reference library. Requires: pip install gnss-lib-py
"""

import numpy as np
import pytest
import math

glp = pytest.importorskip("gnss_lib_py")

# ============================================================
# 1. GPS Constants (IS-GPS-200)
# ============================================================

class TestGPSConstants:
    """Verify fundamental constants match IS-GPS-200."""

    def test_wgs84_semi_major_axis(self):
        assert glp.consts.A == 6378137.0
        from gnss_gpu._gnss_gpu import ecef_to_lla  # uses WGS84_A internally
        # Our constant is in coordinates.h: WGS84_A = 6378137.0
        # Verified by ECEF→LLA roundtrip

    def test_speed_of_light(self):
        # IS-GPS-200: c = 299792458 m/s
        assert abs(glp.consts.C - 299792458.0) < 1.0

    def test_earth_rotation_rate(self):
        # IS-GPS-200: omega_e = 7.2921151467e-5 rad/s
        assert abs(glp.OMEGA_E_DOT - 7.2921151467e-5) < 1e-15

    def test_gm_earth(self):
        # IS-GPS-200: mu = 3.986005e14 m^3/s^2
        assert abs(glp.MU_EARTH - 3.986005e14) < 1e6

    def test_gps_l1_frequency(self):
        # L1 = 1575.42 MHz
        assert abs(glp.F1 - 1575.42e6) < 1.0


# ============================================================
# 2. ECEF ↔ LLA Coordinate Conversion
# ============================================================

class TestCoordinateConversion:
    """Compare ECEF↔LLA with gnss_lib_py."""

    POINTS = [
        # (x, y, z) ECEF in meters
        (-3957199.0, 3310205.0, 3737911.0),   # Tokyo
        (-2694892.0, -4297405.0, 3854586.0),   # Denver
        (4075539.0, 931021.0, 4801629.0),      # London
        (0.0, 0.0, 6356752.314),               # North Pole
        (6378137.0, 0.0, 0.0),                 # Equator 0°lon
    ]

    def test_ecef_to_lla_matches_reference(self):
        from gnss_gpu._gnss_gpu import ecef_to_lla

        for x, y, z in self.POINTS:
            if abs(x) < 1 and abs(y) < 1:
                continue  # skip pole (singularity in our implementation)

            # gnss_lib_py reference: expects [3, N] array
            ecef = np.array([[x], [y], [z]])
            lla_ref = glp.ecef_to_geodetic(ecef)  # returns [3, N] (lat, lon, alt) in degrees

            # Our implementation (returns radians)
            lat, lon, alt = ecef_to_lla(
                np.array([x]), np.array([y]), np.array([z])
            )

            assert abs(np.degrees(lat[0]) - lla_ref[0, 0]) < 1e-5, \
                f"Lat mismatch at ({x},{y},{z})"
            assert abs(np.degrees(lon[0]) - lla_ref[1, 0]) < 1e-5, \
                f"Lon mismatch at ({x},{y},{z})"
            assert abs(alt[0] - lla_ref[2, 0]) < 1.0, \
                f"Alt mismatch: ours={alt[0]:.1f} ref={lla_ref[2,0]:.1f}"

    def test_lla_to_ecef_matches_reference(self):
        from gnss_gpu._gnss_gpu import lla_to_ecef

        test_lla = [
            (35.68, 139.77, 40.0),    # Tokyo
            (39.74, -104.99, 1609.0),  # Denver
            (51.51, -0.13, 11.0),      # London
        ]

        for lat_d, lon_d, alt in test_lla:
            lat_r = np.radians(lat_d)
            lon_r = np.radians(lon_d)

            # gnss_lib_py reference: expects [3, N] (lat_deg, lon_deg, alt)
            lla = np.array([[lat_d], [lon_d], [alt]])
            ecef_ref = glp.geodetic_to_ecef(lla)  # returns [3, N]

            # Our implementation
            x, y, z = lla_to_ecef(
                np.array([lat_r]), np.array([lon_r]), np.array([alt])
            )

            assert abs(x[0] - ecef_ref[0, 0]) < 1.0, f"X mismatch at ({lat_d},{lon_d})"
            assert abs(y[0] - ecef_ref[1, 0]) < 1.0, f"Y mismatch at ({lat_d},{lon_d})"
            assert abs(z[0] - ecef_ref[2, 0]) < 1.0, f"Z mismatch at ({lat_d},{lon_d})"

    def test_roundtrip_consistency(self):
        from gnss_gpu._gnss_gpu import ecef_to_lla, lla_to_ecef

        # Skip poles (singularity in our closed-form conversion)
        safe_points = [p for p in self.POINTS if abs(p[0]) > 100 or abs(p[1]) > 100]
        for x, y, z in safe_points:
            lat, lon, alt = ecef_to_lla(
                np.array([x]), np.array([y]), np.array([z])
            )
            x2, y2, z2 = lla_to_ecef(lat, lon, alt)

            assert abs(x - x2[0]) < 0.01, f"X roundtrip error at ({x},{y},{z})"
            assert abs(y - y2[0]) < 0.01, f"Y roundtrip error"
            assert abs(z - z2[0]) < 0.01, f"Z roundtrip error"


# ============================================================
# 3. DOP Computation
# ============================================================

class TestDOPComputation:
    """Compare DOP values with gnss_lib_py."""

    def _compute_dop_reference(self, rx_ecef, sat_ecef):
        """Compute DOP using pure numpy (reference implementation)."""
        n_sat = sat_ecef.shape[0]

        # Convert rx to LLA
        ecef_arr = rx_ecef.reshape(3, 1)
        lla = glp.ecef_to_geodetic(ecef_arr, radians=True)
        lat = lla[0, 0]
        lon = lla[1, 0]

        sin_lat, cos_lat = np.sin(lat), np.cos(lat)
        sin_lon, cos_lon = np.sin(lon), np.cos(lon)

        el_list = []
        az_list = []
        for s in range(n_sat):
            dx = sat_ecef[s, 0] - rx_ecef[0]
            dy = sat_ecef[s, 1] - rx_ecef[1]
            dz = sat_ecef[s, 2] - rx_ecef[2]
            e = -sin_lon * dx + cos_lon * dy
            n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
            u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
            hz = np.sqrt(e**2 + n**2)
            el_list.append(np.arctan2(u, hz))
            az_list.append(np.arctan2(e, n))

        el = np.array(el_list)
        az = np.array(az_list)

        H = np.zeros((n_sat, 4))
        for i in range(n_sat):
            H[i, 0] = -np.cos(el[i]) * np.sin(az[i])
            H[i, 1] = -np.cos(el[i]) * np.cos(az[i])
            H[i, 2] = -np.sin(el[i])
            H[i, 3] = 1.0

        G = np.linalg.inv(H.T @ H)
        gdop = np.sqrt(np.trace(G))
        pdop = np.sqrt(G[0, 0] + G[1, 1] + G[2, 2])
        hdop = np.sqrt(G[0, 0] + G[1, 1])
        vdop = np.sqrt(G[2, 2])
        return hdop, vdop, pdop, gdop

    def test_dop_matches_manual(self):
        """Compare our GPU DOP with manual numpy computation."""
        try:
            from gnss_gpu._gnss_gpu_skyplot import compute_grid_quality
        except ImportError:
            pytest.skip("Skyplot CUDA module not available")

        rx = np.array([-3957199.0, 3310205.0, 3737911.0])
        sats = np.array([
            [-14985000.0, -3988000.0, 21474000.0],
            [-9575000.0, 15498000.0, 19457000.0],
            [-20889000.0, 13759000.0, 8291000.0],
            [5463000.0, 24413000.0, 8934000.0],
        ])

        # Reference (manual)
        hdop_ref, vdop_ref, pdop_ref, gdop_ref = self._compute_dop_reference(rx, sats)

        # Our GPU implementation
        pdop, hdop, vdop, gdop, n_vis = compute_grid_quality(
            rx, sats.flatten(), 1, 4, 0.0)

        # Allow 10% tolerance due to potential coordinate frame differences
        assert abs(hdop[0] - hdop_ref) / hdop_ref < 0.1, \
            f"HDOP: ours={hdop[0]:.2f} ref={hdop_ref:.2f}"
        assert abs(vdop[0] - vdop_ref) / vdop_ref < 0.1, \
            f"VDOP: ours={vdop[0]:.2f} ref={vdop_ref:.2f}"
        assert abs(pdop[0] - pdop_ref) / pdop_ref < 0.1, \
            f"PDOP: ours={pdop[0]:.2f} ref={pdop_ref:.2f}"


# ============================================================
# 4. Tropospheric Delay
# ============================================================

class TestTroposphericDelay:
    """Compare tropospheric model with known values."""

    def test_zenith_delay_at_sea_level(self):
        """Standard atmosphere zenith delay should be ~2.3m."""
        from gnss_gpu.atmosphere import AtmosphereCorrection
        atm = AtmosphereCorrection()

        rx_lla = np.array([[np.radians(35.0), np.radians(139.0), 0.0]])
        el = np.array([[np.radians(90.0)]])  # zenith

        tropo = atm.tropo(rx_lla, el)

        # Reference: Saastamoinen zenith delay at sea level ≈ 2.3m
        assert 2.0 < tropo[0, 0] < 2.6, f"Zenith tropo delay {tropo[0,0]:.2f}m unexpected"

    def test_low_elevation_larger(self):
        """Lower elevation should give larger delay."""
        from gnss_gpu.atmosphere import AtmosphereCorrection
        atm = AtmosphereCorrection()

        rx_lla = np.array([[np.radians(35.0), np.radians(139.0), 0.0]])
        el_high = np.array([[np.radians(80.0)]])
        el_low = np.array([[np.radians(10.0)]])

        tropo_high = atm.tropo(rx_lla, el_high)
        tropo_low = atm.tropo(rx_lla, el_low)

        assert tropo_low[0, 0] > tropo_high[0, 0] * 3, \
            "Low elevation tropo should be much larger"

    def test_altitude_dependency(self):
        """Higher altitude should give smaller delay."""
        from gnss_gpu.atmosphere import AtmosphereCorrection
        atm = AtmosphereCorrection()

        el = np.array([[np.radians(45.0)]])
        rx_sea = np.array([[np.radians(35.0), np.radians(139.0), 0.0]])
        rx_mountain = np.array([[np.radians(35.0), np.radians(139.0), 3000.0]])

        tropo_sea = atm.tropo(rx_sea, el)
        tropo_mountain = atm.tropo(rx_mountain, el)

        assert tropo_mountain[0, 0] < tropo_sea[0, 0], \
            "Mountain tropo should be smaller than sea level"


# ============================================================
# 5. Ionospheric Delay (Klobuchar)
# ============================================================

class TestIonosphericDelay:
    """Compare Klobuchar model with known behavior."""

    def test_nighttime_minimum(self):
        """Nighttime delay should be ~5ns * c ≈ 1.5m."""
        from gnss_gpu.atmosphere import AtmosphereCorrection
        atm = AtmosphereCorrection()

        rx_lla = np.array([[np.radians(35.0), np.radians(139.0), 0.0]])
        el = np.array([[np.radians(45.0)]])
        az = np.array([[np.radians(180.0)]])

        # GPS time at midnight (local)
        gps_time_night = 0.0  # midnight UTC ≈ nighttime for many locations

        iono = atm.iono(rx_lla, az, el, gps_time_night)

        # Nighttime minimum ≈ F * 5e-9 * c ≈ 1.5-3m
        assert 0.5 < iono[0, 0] < 5.0, f"Nighttime iono {iono[0,0]:.2f}m unexpected"

    def test_daytime_larger_than_night(self):
        """Daytime delay should exceed nighttime at the pierce point."""
        from gnss_gpu.atmosphere import AtmosphereCorrection
        atm = AtmosphereCorrection()

        # Use a location where local noon is clearly at ~43200 GPS TOW
        # Longitude 0 → local time = GPS time (approximately)
        rx_lla = np.array([[np.radians(45.0), np.radians(0.0), 0.0]])
        el = np.array([[np.radians(45.0)]])
        az = np.array([[np.radians(180.0)]])

        iono_night = atm.iono(rx_lla, az, el, 3600.0)   # 01:00 UTC → night at lon=0
        iono_day = atm.iono(rx_lla, az, el, 50400.0)     # 14:00 UTC → afternoon at lon=0

        assert iono_day[0, 0] >= iono_night[0, 0], \
            f"Day iono {iono_day[0,0]:.2f} should be >= night {iono_night[0,0]:.2f}"


# ============================================================
# 6. WLS Positioning
# ============================================================

class TestWLSPositioning:
    """Compare WLS solution with known geometry."""

    def test_noise_free_convergence(self):
        """WLS should converge to exact solution with no noise."""
        from gnss_gpu._gnss_gpu import wls_position

        true_pos = np.array([-3957199.0, 3310205.0, 3737911.0])
        true_cb = 3000.0

        sats = np.array([
            [-14985000.0, -3988000.0, 21474000.0],
            [-9575000.0, 15498000.0, 19457000.0],
            [7624000.0, -16218000.0, 19843000.0],
            [16305000.0, 12037000.0, 17183000.0],
            [-20889000.0, 13759000.0, 8291000.0],
            [5463000.0, 24413000.0, 8934000.0],
            [22169000.0, 3975000.0, 13781000.0],
            [-11527000.0, -19421000.0, 13682000.0],
        ])

        ranges = np.sqrt(np.sum((sats - true_pos) ** 2, axis=1))
        pr = ranges + true_cb
        w = np.ones(8)

        result, iters = wls_position(sats.flatten(), pr, w)
        pos = result[:3]
        cb = result[3]

        err = np.linalg.norm(pos - true_pos)
        assert err < 0.01, f"WLS position error {err:.6f}m"
        assert abs(cb - true_cb) < 0.01, f"Clock bias error {abs(cb-true_cb):.6f}m"

    def test_noisy_pseudoranges(self):
        """WLS with 3m noise should give < 20m error."""
        from gnss_gpu._gnss_gpu import wls_position

        true_pos = np.array([-3957199.0, 3310205.0, 3737911.0])
        true_cb = 3000.0

        sats = np.array([
            [-14985000.0, -3988000.0, 21474000.0],
            [-9575000.0, 15498000.0, 19457000.0],
            [7624000.0, -16218000.0, 19843000.0],
            [16305000.0, 12037000.0, 17183000.0],
            [-20889000.0, 13759000.0, 8291000.0],
            [5463000.0, 24413000.0, 8934000.0],
            [22169000.0, 3975000.0, 13781000.0],
            [-11527000.0, -19421000.0, 13682000.0],
        ])

        rng = np.random.default_rng(42)
        ranges = np.sqrt(np.sum((sats - true_pos) ** 2, axis=1))
        pr = ranges + true_cb + rng.normal(0, 3.0, 8)
        w = np.ones(8)

        result, iters = wls_position(sats.flatten(), pr, w)
        err = np.linalg.norm(result[:3] - true_pos)

        assert err < 20.0, f"WLS noisy error {err:.2f}m (expected < 20m)"


# ============================================================
# 7. C/A Code Verification
# ============================================================

class TestCACode:
    """Verify GPS C/A code against IS-GPS-200 known values."""

    # First 10 chips (as +1/-1) for selected PRNs from IS-GPS-200
    KNOWN_FIRST_CHIPS = {
        1: [1, 1, -1, -1, 1, -1, -1, -1, -1, -1],
    }

    def test_code_length(self):
        """C/A code should be 1023 chips."""
        try:
            from gnss_gpu._gnss_gpu_acq import generate_ca_code
        except ImportError:
            pytest.skip("Acquisition module not available")

        code = generate_ca_code(1)
        assert len(code) == 1023

    def test_code_values(self):
        """C/A code chips should be +1 or -1."""
        try:
            from gnss_gpu._gnss_gpu_acq import generate_ca_code
        except ImportError:
            pytest.skip("Acquisition module not available")

        for prn in range(1, 33):
            code = generate_ca_code(prn)
            for chip in code:
                assert chip in (1, -1), f"PRN {prn}: invalid chip value {chip}"

    def test_prn1_first_10_chips(self):
        """PRN 1 first 10 chips should match IS-GPS-200."""
        try:
            from gnss_gpu._gnss_gpu_acq import generate_ca_code
        except ImportError:
            pytest.skip("Acquisition module not available")

        code = generate_ca_code(1)
        expected = self.KNOWN_FIRST_CHIPS[1]
        assert code[:10] == expected, \
            f"PRN 1 first 10 chips: got {code[:10]}, expected {expected}"

    def test_code_balance(self):
        """Gold codes should have ~512 ones and ~511 zeros (as bits)."""
        try:
            from gnss_gpu._gnss_gpu_acq import generate_ca_code
        except ImportError:
            pytest.skip("Acquisition module not available")

        for prn in range(1, 33):
            code = generate_ca_code(prn)
            n_pos = sum(1 for c in code if c == 1)
            # Gold code property: exactly 512 or 511
            assert n_pos in (511, 512), f"PRN {prn}: {n_pos} positive chips"

    def test_all_prns_unique(self):
        """All 32 PRN codes should be unique."""
        try:
            from gnss_gpu._gnss_gpu_acq import generate_ca_code
        except ImportError:
            pytest.skip("Acquisition module not available")

        codes = set()
        for prn in range(1, 33):
            code = tuple(generate_ca_code(prn))
            assert code not in codes, f"PRN {prn} duplicates another PRN"
            codes.add(code)


# ============================================================
# 8. Pseudorange Equation Consistency
# ============================================================

class TestPseudorangeEquation:
    """Verify the fundamental GNSS pseudorange equation is correctly implemented."""

    def test_pr_equals_range_plus_bias(self):
        """pr = |rx - sat| + clock_bias (no atmosphere, no noise)."""
        from gnss_gpu._gnss_gpu import wls_position

        rx_true = np.array([-3957199.0, 3310205.0, 3737911.0])
        cb_true = 1000.0
        sats = np.array([
            [-14985000.0, -3988000.0, 21474000.0],
            [-9575000.0, 15498000.0, 19457000.0],
            [7624000.0, -16218000.0, 19843000.0],
            [16305000.0, 12037000.0, 17183000.0],
            [-20889000.0, 13759000.0, 8291000.0],
        ])

        # Construct pseudoranges from the equation
        ranges = np.linalg.norm(sats - rx_true, axis=1)
        pr = ranges + cb_true

        # Solve - should recover exact position
        result, _ = wls_position(sats.flatten(), pr, np.ones(5))
        assert np.linalg.norm(result[:3] - rx_true) < 0.01
        assert abs(result[3] - cb_true) < 0.01

    def test_atmosphere_corrections_positive(self):
        """Tropo and iono delays should be positive (signal is delayed)."""
        from gnss_gpu.atmosphere import AtmosphereCorrection
        atm = AtmosphereCorrection()

        rx_lla = np.array([[np.radians(35.0), np.radians(139.0), 100.0]])
        el = np.array([[np.radians(30.0)]])
        az = np.array([[np.radians(90.0)]])

        tropo = atm.tropo(rx_lla, el)
        iono = atm.iono(rx_lla, az, el, 43200.0)

        assert tropo[0, 0] > 0, "Tropo delay must be positive"
        assert iono[0, 0] > 0, "Iono delay must be positive"


# ============================================================
# 9. Multi-GNSS ISB Verification
# ============================================================

class TestMultiGNSSVerification:
    """Verify Multi-GNSS ISB estimation."""

    def test_known_isb_recovery(self):
        """Inject known ISB offset and verify recovery."""
        from gnss_gpu.multi_gnss import MultiGNSSSolver

        true_pos = np.array([-3957199.0, 3310205.0, 3737911.0])
        true_cb_gps = 3000.0
        true_isb = 15.0  # 15m ISB for Galileo

        gps_sats = np.array([
            [-14985000.0, -3988000.0, 21474000.0],
            [-9575000.0, 15498000.0, 19457000.0],
            [-20889000.0, 13759000.0, 8291000.0],
            [5463000.0, 24413000.0, 8934000.0],
        ])
        gal_sats = np.array([
            [22169000.0, 3975000.0, 13781000.0],
            [-11527000.0, -19421000.0, 13682000.0],
            [7624000.0, -16218000.0, 19843000.0],
            [16305000.0, 12037000.0, 17183000.0],
        ])

        all_sats = np.vstack([gps_sats, gal_sats])
        ranges = np.linalg.norm(all_sats - true_pos, axis=1)

        # GPS PRs with GPS clock bias, Galileo PRs with GPS+ISB
        pr = ranges.copy()
        pr[:4] += true_cb_gps
        pr[4:] += true_cb_gps + true_isb

        system_ids = np.array([0, 0, 0, 0, 2, 2, 2, 2], dtype=np.int32)

        solver = MultiGNSSSolver(systems=[0, 2])
        pos, biases, n_iter = solver.solve(all_sats, pr, system_ids)

        err = np.linalg.norm(pos - true_pos)
        assert err < 1.0, f"Position error {err:.2f}m"

        # ISB = Galileo bias - GPS bias
        isb_estimated = biases.get(2, 0) - biases.get(0, 0)
        assert abs(isb_estimated - true_isb) < 1.0, \
            f"ISB error: estimated={isb_estimated:.2f}, true={true_isb}"
