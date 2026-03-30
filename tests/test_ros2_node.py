"""Tests for the ROS2 GNSS GPU positioning node.

All tests that require rclpy are skipped if ROS2 is not installed.
Pure-Python helpers (ECEF-to-LLA conversion, config generation) are tested
unconditionally.
"""

import math
import struct
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


# ---------------------------------------------------------------------------
# Detect ROS2 availability
# ---------------------------------------------------------------------------
try:
    import rclpy
    from sensor_msgs.msg import NavSatFix, PointCloud2
    _HAS_ROS2 = True
except ImportError:
    _HAS_ROS2 = False


# ---------------------------------------------------------------------------
# Import module-level helpers that do NOT require rclpy at import time
# ---------------------------------------------------------------------------
from gnss_gpu.ros2.gnss_node import ecef_to_lla


class TestECEFtoLLA(unittest.TestCase):
    """Test the pure-Python ECEF-to-LLA conversion."""

    def test_origin_on_equator(self):
        """A point on the equator at zero longitude should give lat=0, lon=0."""
        # Surface of WGS84 ellipsoid at (lat=0, lon=0)
        a = 6378137.0
        lat, lon, alt = ecef_to_lla(a, 0.0, 0.0)
        self.assertAlmostEqual(lat, 0.0, places=6)
        self.assertAlmostEqual(lon, 0.0, places=6)
        self.assertAlmostEqual(alt, 0.0, delta=1.0)

    def test_north_pole(self):
        """A point at the north pole."""
        b = 6356752.314245
        lat, lon, alt = ecef_to_lla(0.0, 0.0, b)
        self.assertAlmostEqual(lat, 90.0, places=4)
        self.assertAlmostEqual(alt, 0.0, delta=1.0)

    def test_tokyo(self):
        """Known coordinates for Tokyo (approximately)."""
        # Tokyo: lat ~35.6762, lon ~139.6503, alt ~40 m
        lat_ref = 35.6762
        lon_ref = 139.6503
        alt_ref = 40.0

        lat_rad = math.radians(lat_ref)
        lon_rad = math.radians(lon_ref)
        a = 6378137.0
        e2 = 6.69437999014e-3
        sin_lat = math.sin(lat_rad)
        cos_lat = math.cos(lat_rad)
        N = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)

        x = (N + alt_ref) * cos_lat * math.cos(lon_rad)
        y = (N + alt_ref) * cos_lat * math.sin(lon_rad)
        z = (N * (1.0 - e2) + alt_ref) * sin_lat

        lat, lon, alt = ecef_to_lla(x, y, z)
        self.assertAlmostEqual(lat, lat_ref, places=4)
        self.assertAlmostEqual(lon, lon_ref, places=4)
        self.assertAlmostEqual(alt, alt_ref, delta=1.0)

    def test_southern_hemisphere(self):
        """Negative latitude for a point in the southern hemisphere."""
        lat_ref = -33.8688  # Sydney
        lon_ref = 151.2093
        alt_ref = 0.0

        lat_rad = math.radians(lat_ref)
        lon_rad = math.radians(lon_ref)
        a = 6378137.0
        e2 = 6.69437999014e-3
        sin_lat = math.sin(lat_rad)
        cos_lat = math.cos(lat_rad)
        N = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)

        x = (N + alt_ref) * cos_lat * math.cos(lon_rad)
        y = (N + alt_ref) * cos_lat * math.sin(lon_rad)
        z = (N * (1.0 - e2) + alt_ref) * sin_lat

        lat, lon, alt = ecef_to_lla(x, y, z)
        self.assertAlmostEqual(lat, lat_ref, places=4)
        self.assertAlmostEqual(lon, lon_ref, places=4)


# ---------------------------------------------------------------------------
# Tests that require ROS2
# ---------------------------------------------------------------------------
@unittest.skipUnless(_HAS_ROS2, "ROS2 (rclpy) is not available")
class TestNavSatFixCreation(unittest.TestCase):
    """Test NavSatFix message creation."""

    def test_basic_fix(self):
        from gnss_gpu.ros2.gnss_node import create_navsatfix
        msg = create_navsatfix(35.6762, 139.6503, 40.0)
        self.assertIsInstance(msg, NavSatFix)
        self.assertAlmostEqual(msg.latitude, 35.6762, places=4)
        self.assertAlmostEqual(msg.longitude, 139.6503, places=4)
        self.assertAlmostEqual(msg.altitude, 40.0, places=1)

    def test_with_covariance(self):
        from gnss_gpu.ros2.gnss_node import create_navsatfix
        cov = np.eye(3).ravel()
        msg = create_navsatfix(0.0, 0.0, 0.0, covariance=cov)
        self.assertEqual(
            msg.position_covariance_type,
            NavSatFix.COVARIANCE_TYPE_KNOWN)
        self.assertEqual(len(msg.position_covariance), 9)


@unittest.skipUnless(_HAS_ROS2, "ROS2 (rclpy) is not available")
class TestPointCloud2Creation(unittest.TestCase):
    """Test PointCloud2 message creation."""

    def test_basic_cloud(self):
        from gnss_gpu.ros2.gnss_node import create_pointcloud2
        particles = np.random.randn(100, 3).astype(np.float64)
        # Shift to ECEF-like coordinates
        particles[:, 0] += 6378137.0
        msg = create_pointcloud2(particles, frame_id="map")
        self.assertIsInstance(msg, PointCloud2)
        self.assertEqual(msg.width, 100)
        self.assertEqual(msg.height, 1)
        self.assertTrue(msg.is_dense)
        self.assertEqual(len(msg.fields), 3)

    def test_single_point(self):
        from gnss_gpu.ros2.gnss_node import create_pointcloud2
        particles = np.array([[1.0, 2.0, 3.0]])
        msg = create_pointcloud2(particles)
        self.assertEqual(msg.width, 1)
        # Verify data can be unpacked
        x, y, z = struct.unpack_from('fff', msg.data, 0)
        # Single point: local offset is (0, 0, 0)
        self.assertAlmostEqual(x, 0.0, places=4)
        self.assertAlmostEqual(y, 0.0, places=4)
        self.assertAlmostEqual(z, 0.0, places=4)


@unittest.skipUnless(_HAS_ROS2, "ROS2 (rclpy) is not available")
class TestGNSSNodeCreation(unittest.TestCase):
    """Test node instantiation with mocked particle filter."""

    def test_node_requires_ros2(self):
        """Verify the node raises if rclpy is missing."""
        from gnss_gpu.ros2 import gnss_node as gn

        original = gn.HAS_ROS2
        try:
            gn.HAS_ROS2 = False
            with self.assertRaises(RuntimeError):
                gn.GNSSPositioningNode()
        finally:
            gn.HAS_ROS2 = original


class TestGNSSNodeWithoutROS2(unittest.TestCase):
    """Test node behaviour when ROS2 is not available (always runs)."""

    def test_has_ros2_flag_is_bool(self):
        from gnss_gpu.ros2.gnss_node import HAS_ROS2
        self.assertIsInstance(HAS_ROS2, bool)

    def test_node_raises_without_ros2(self):
        """If ROS2 is missing, GNSSPositioningNode should raise RuntimeError."""
        from gnss_gpu.ros2 import gnss_node as gn
        if gn.HAS_ROS2:
            self.skipTest("ROS2 is available; this test is for when it is not")
        with self.assertRaises(RuntimeError):
            gn.GNSSPositioningNode()


# ---------------------------------------------------------------------------
# RViz config tests (no ROS2 dependency)
# ---------------------------------------------------------------------------
class TestRVizConfig(unittest.TestCase):
    """Test the RViz configuration helper."""

    def test_get_config_string(self):
        from gnss_gpu.ros2.rviz_config import get_rviz_config_string
        config = get_rviz_config_string()
        self.assertIn('/gnss/particles', config)
        self.assertIn('/gnss/fix_pf', config)
        self.assertIn('/gnss/skyplot', config)

    def test_generate_config_file(self):
        import tempfile
        import os
        from gnss_gpu.ros2.rviz_config import generate_rviz_config
        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_rviz_config(
                output_path=os.path.join(tmpdir, 'test.rviz'))
            self.assertTrue(os.path.isfile(path))
            with open(path) as f:
                content = f.read()
            self.assertIn('/gnss/particles', content)


if __name__ == '__main__':
    unittest.main()
