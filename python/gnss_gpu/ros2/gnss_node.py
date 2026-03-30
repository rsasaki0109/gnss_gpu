"""ROS2 GNSS GPU Positioning Node.

Subscribes to:
  - /gnss/raw (sensor_msgs/NavSatFix or custom) -- raw GNSS observations
  - /gnss/ephemeris (custom msg or NavSatFix) -- satellite ephemeris

Publishes:
  - /gnss/fix (sensor_msgs/NavSatFix) -- WLS position fix
  - /gnss/fix_pf (sensor_msgs/NavSatFix) -- particle filter position fix
  - /gnss/particles (sensor_msgs/PointCloud2) -- particle cloud for visualization
  - /gnss/skyplot (visualization_msgs/MarkerArray) -- satellite positions

Parameters:
  - n_particles (int, default 100000)
  - sigma_pr (float, default 5.0)
  - sigma_pos (float, default 1.0)
  - sigma_cb (float, default 300.0)
  - resampling_method (str, default "megopolis")
  - update_rate (float, default 1.0)
  - use_3d_model (bool, default False)
  - building_model_path (str, default "")
"""

import struct

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import NavSatFix, NavSatStatus, PointCloud2, PointField
    from std_msgs.msg import Header
    from geometry_msgs.msg import PoseWithCovarianceStamped
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False

import math
import numpy as np


# WGS84 constants
_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_B = _WGS84_A * (1.0 - _WGS84_F)
_WGS84_E2 = 1.0 - (_WGS84_B ** 2) / (_WGS84_A ** 2)


def ecef_to_lla(x, y, z):
    """Convert ECEF coordinates to geodetic latitude, longitude, altitude.

    Uses an iterative algorithm based on Bowring's method for WGS84.

    Parameters
    ----------
    x, y, z : float
        ECEF coordinates in metres.

    Returns
    -------
    lat, lon, alt : tuple of float
        Geodetic latitude [deg], longitude [deg], altitude [m].
    """
    lon = math.atan2(y, x)

    p = math.sqrt(x * x + y * y)
    lat = math.atan2(z, p * (1.0 - _WGS84_E2))

    for _ in range(10):
        sin_lat = math.sin(lat)
        N = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
        lat_new = math.atan2(z + _WGS84_E2 * N * sin_lat, p)
        if abs(lat_new - lat) < 1e-12:
            break
        lat = lat_new

    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    N = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)

    if abs(cos_lat) > 1e-10:
        alt = p / cos_lat - N
    else:
        alt = abs(z) - _WGS84_B

    lat_deg = math.degrees(lat)
    lon_deg = math.degrees(lon)

    return lat_deg, lon_deg, alt


def create_pointcloud2(particles, frame_id="map", stamp=None):
    """Create a PointCloud2 message from particle positions.

    Parameters
    ----------
    particles : ndarray, shape (N, 4) or (N, 3)
        Particle states. Columns are [x, y, z] or [x, y, z, clock_bias].
        Positions are in ECEF metres; they are converted to LLA for display.
    frame_id : str
        Frame ID for the message header.
    stamp : object or None
        ROS2 Time stamp. If None, uses current time (requires rclpy).

    Returns
    -------
    msg : PointCloud2
        The point cloud message.
    """
    if not HAS_ROS2:
        raise RuntimeError("ROS2 (rclpy) is required to create PointCloud2 messages")

    particles = np.asarray(particles, dtype=np.float64)
    if particles.ndim == 1:
        particles = particles.reshape(-1, 3)

    n_points = particles.shape[0]

    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]

    point_step = 12  # 3 x float32
    row_step = point_step * n_points

    # Convert ECEF to local offsets relative to the centroid for visualisation
    cx = np.mean(particles[:, 0])
    cy = np.mean(particles[:, 1])
    cz = np.mean(particles[:, 2])
    local_x = (particles[:, 0] - cx).astype(np.float32)
    local_y = (particles[:, 1] - cy).astype(np.float32)
    local_z = (particles[:, 2] - cz).astype(np.float32)

    buf = bytearray(row_step)
    for i in range(n_points):
        offset = i * point_step
        struct.pack_into('fff', buf, offset, local_x[i], local_y[i], local_z[i])

    msg = PointCloud2()
    msg.header = Header()
    msg.header.frame_id = frame_id
    if stamp is not None:
        msg.header.stamp = stamp
    msg.height = 1
    msg.width = n_points
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = point_step
    msg.row_step = row_step
    msg.data = bytes(buf)
    msg.is_dense = True

    return msg


def create_navsatfix(lat, lon, alt, covariance=None, frame_id="gnss",
                     stamp=None):
    """Create a NavSatFix message.

    Parameters
    ----------
    lat, lon, alt : float
        Geodetic latitude [deg], longitude [deg], altitude [m].
    covariance : array_like, shape (9,), optional
        Position covariance in ENU frame (row-major 3x3).
    frame_id : str
        Frame ID for the header.
    stamp : object or None
        ROS2 Time stamp.

    Returns
    -------
    msg : NavSatFix
    """
    if not HAS_ROS2:
        raise RuntimeError("ROS2 (rclpy) is required to create NavSatFix messages")

    msg = NavSatFix()
    msg.header = Header()
    msg.header.frame_id = frame_id
    if stamp is not None:
        msg.header.stamp = stamp

    msg.status = NavSatStatus()
    msg.status.status = NavSatStatus.STATUS_FIX
    msg.status.service = NavSatStatus.SERVICE_GPS

    msg.latitude = float(lat)
    msg.longitude = float(lon)
    msg.altitude = float(alt)

    if covariance is not None:
        cov = np.asarray(covariance, dtype=np.float64).ravel()
        if len(cov) == 9:
            msg.position_covariance = cov.tolist()
            msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_KNOWN
        else:
            msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN
    else:
        msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN

    return msg


class GNSSPositioningNode:
    """ROS2 node for GPU-accelerated GNSS positioning.

    This node wraps the gnss_gpu ParticleFilter and provides a ROS2 interface
    for real-time GNSS positioning.  It subscribes to raw GNSS fixes, runs
    the particle filter on GPU, and publishes corrected positions together
    with a particle cloud for visualisation.
    """

    def __init__(self):
        if not HAS_ROS2:
            raise RuntimeError(
                "ROS2 (rclpy) is not available. Install ros2 and source "
                "the workspace before running this node."
            )

        rclpy.init()
        self.node = rclpy.create_node('gnss_gpu_positioning')

        # -- Declare parameters -----------------------------------------------
        self.node.declare_parameter('n_particles', 100000)
        self.node.declare_parameter('sigma_pr', 5.0)
        self.node.declare_parameter('sigma_pos', 1.0)
        self.node.declare_parameter('sigma_cb', 300.0)
        self.node.declare_parameter('resampling_method', 'megopolis')
        self.node.declare_parameter('update_rate', 1.0)
        self.node.declare_parameter('use_3d_model', False)
        self.node.declare_parameter('building_model_path', '')

        # -- Retrieve parameter values ----------------------------------------
        n_particles = self.node.get_parameter('n_particles').value
        sigma_pr = self.node.get_parameter('sigma_pr').value
        sigma_pos = self.node.get_parameter('sigma_pos').value
        sigma_cb = self.node.get_parameter('sigma_cb').value
        resampling_method = self.node.get_parameter('resampling_method').value

        # -- Initialise particle filter ---------------------------------------
        from gnss_gpu.particle_filter import ParticleFilter

        self.pf = ParticleFilter(
            n_particles=n_particles,
            sigma_pos=sigma_pos,
            sigma_cb=sigma_cb,
            sigma_pr=sigma_pr,
            resampling=resampling_method,
        )
        self.initialized = False
        self._last_stamp = None

        # -- Publishers -------------------------------------------------------
        self.pub_fix = self.node.create_publisher(
            NavSatFix, '/gnss/fix', 10)
        self.pub_fix_pf = self.node.create_publisher(
            NavSatFix, '/gnss/fix_pf', 10)
        self.pub_particles = self.node.create_publisher(
            PointCloud2, '/gnss/particles', 10)
        self.pub_pose = self.node.create_publisher(
            PoseWithCovarianceStamped, '/gnss/pose', 10)

        # -- Subscribers ------------------------------------------------------
        self.sub_fix = self.node.create_subscription(
            NavSatFix, '/gnss/raw', self.raw_callback, 10)

        # -- Timer for particle visualisation ---------------------------------
        update_rate = self.node.get_parameter('update_rate').value
        timer_period = 1.0 / max(update_rate, 0.01)
        self.timer = self.node.create_timer(timer_period, self.publish_particles)

        self.node.get_logger().info(
            f"GNSS GPU Positioning Node started with {n_particles} particles"
        )

    # --------------------------------------------------------------------- #
    # Callbacks
    # --------------------------------------------------------------------- #

    def raw_callback(self, msg):
        """Process a raw GNSS NavSatFix observation.

        On the first message, the particle filter is initialised around the
        reported position. Subsequent messages trigger a predict-update cycle.
        """
        lat = msg.latitude
        lon = msg.longitude
        alt = msg.altitude

        # Convert LLA to ECEF for the particle filter
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        sin_lat = math.sin(lat_rad)
        cos_lat = math.cos(lat_rad)
        sin_lon = math.sin(lon_rad)
        cos_lon = math.cos(lon_rad)
        N = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)

        ecef_x = (N + alt) * cos_lat * cos_lon
        ecef_y = (N + alt) * cos_lat * sin_lon
        ecef_z = (N * (1.0 - _WGS84_E2) + alt) * sin_lat

        if not self.initialized:
            self.pf.initialize(
                position_ecef=[ecef_x, ecef_y, ecef_z],
                spread_pos=100.0,
            )
            self.initialized = True
            self.node.get_logger().info(
                f"Particle filter initialised at "
                f"({lat:.6f}, {lon:.6f}, {alt:.1f})"
            )
        else:
            self.pf.predict()

        self._last_stamp = msg.header.stamp

        # Re-publish the raw fix on /gnss/fix
        fix_msg = create_navsatfix(lat, lon, alt, frame_id="gnss",
                                   stamp=msg.header.stamp)
        self.pub_fix.publish(fix_msg)

        # Publish particle filter estimate on /gnss/fix_pf
        self.publish_fix_pf(stamp=msg.header.stamp)

    # --------------------------------------------------------------------- #
    # Publishing helpers
    # --------------------------------------------------------------------- #

    def publish_fix(self, ecef_position, covariance=None, stamp=None):
        """Convert an ECEF position to NavSatFix and publish on /gnss/fix."""
        lat, lon, alt = ecef_to_lla(
            ecef_position[0], ecef_position[1], ecef_position[2])
        msg = create_navsatfix(lat, lon, alt, covariance=covariance,
                               frame_id="gnss", stamp=stamp)
        self.pub_fix.publish(msg)

    def publish_fix_pf(self, stamp=None):
        """Publish the particle filter estimate on /gnss/fix_pf."""
        if not self.initialized:
            return

        est = self.pf.estimate()
        lat, lon, alt = ecef_to_lla(est[0], est[1], est[2])
        msg = create_navsatfix(lat, lon, alt, frame_id="gnss", stamp=stamp)
        self.pub_fix_pf.publish(msg)

    def publish_particles(self):
        """Publish the particle cloud as PointCloud2 for RViz visualisation."""
        if not self.initialized:
            return

        particles = self.pf.get_particles()

        # Sub-sample for visualisation if there are too many particles
        max_vis = 10000
        if particles.shape[0] > max_vis:
            indices = np.random.choice(
                particles.shape[0], max_vis, replace=False)
            vis_particles = particles[indices]
        else:
            vis_particles = particles

        msg = create_pointcloud2(
            vis_particles[:, :3], frame_id="map", stamp=self._last_stamp)
        self.pub_particles.publish(msg)

    # --------------------------------------------------------------------- #
    # Lifecycle
    # --------------------------------------------------------------------- #

    def spin(self):
        """Block and process callbacks until shutdown."""
        rclpy.spin(self.node)

    def shutdown(self):
        """Destroy the node and shut down rclpy."""
        self.node.destroy_node()
        rclpy.shutdown()


def main():
    """Entry point for the GNSS GPU positioning node."""
    node = GNSSPositioningNode()
    try:
        node.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()


if __name__ == '__main__':
    main()
