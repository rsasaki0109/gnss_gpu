"""ROS 2 node: robust GNSS fix filtering for outdoor robots.

Subscribes to ``sensor_msgs/NavSatFix`` and republishes a spike-gated,
Kalman-smoothed fix plus an RViz-friendly ``nav_msgs/Path`` in a local
East/North frame anchored at the first fix. The filter math lives in
:mod:`gnss_gpu_ros.filters` (NumPy only) and is the causal port of the
trajectory post-processing stack validated on GSDC2023.
"""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix

from gnss_gpu_ros.filters import NavSatTrajectoryFilter


class RobustNavSatFilter(Node):
    def __init__(self) -> None:
        super().__init__("robust_navsat_filter")
        self.declare_parameter("hampel_window", 21)
        self.declare_parameter("hampel_k", 2.5)
        self.declare_parameter("kalman_sigma_a", 1.0)
        self.declare_parameter("kalman_sigma_z", 1.0)
        self.declare_parameter("use_hampel", True)
        self.declare_parameter("use_kalman", True)
        self.declare_parameter("max_gap_s", 30.0)
        self.declare_parameter("path_frame_id", "map")
        self.declare_parameter("path_max_poses", 2000)

        self._filter = NavSatTrajectoryFilter(
            hampel_window=int(self.get_parameter("hampel_window").value),
            hampel_k=float(self.get_parameter("hampel_k").value),
            kalman_sigma_a=float(self.get_parameter("kalman_sigma_a").value),
            kalman_sigma_z=float(self.get_parameter("kalman_sigma_z").value),
            use_hampel=bool(self.get_parameter("use_hampel").value),
            use_kalman=bool(self.get_parameter("use_kalman").value),
            max_gap_s=float(self.get_parameter("max_gap_s").value),
        )
        self._path = Path()
        self._path.header.frame_id = str(self.get_parameter("path_frame_id").value)
        self._path_max = int(self.get_parameter("path_max_poses").value)
        self._n_outliers = 0
        self._warned_zero_stamp = False

        self._pub_fix = self.create_publisher(NavSatFix, "fix_filtered", 10)
        self._pub_path = self.create_publisher(Path, "path_filtered", 10)
        self._sub = self.create_subscription(NavSatFix, "fix", self._on_fix, 50)
        self.get_logger().info(
            "robust_navsat_filter ready: fix -> fix_filtered, path_filtered"
        )

    def _on_fix(self, msg: NavSatFix) -> None:
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if t == 0.0:
            # Driver left header.stamp unset; fall back to receive time so the
            # Kalman stage still sees monotonic timestamps.
            if not self._warned_zero_stamp:
                self._warned_zero_stamp = True
                self.get_logger().warn(
                    "incoming NavSatFix has zero header.stamp; using receive time"
                )
            t = self.get_clock().now().nanoseconds * 1e-9
        try:
            lat, lon, east, north, outlier = self._filter.update(
                t, msg.latitude, msg.longitude
            )
        except ValueError as exc:
            self.get_logger().warn(f"dropping invalid NavSatFix: {exc}")
            return
        if outlier:
            self._n_outliers += 1
            self.get_logger().debug(
                f"fix gated as outlier (#{self._n_outliers}): "
                f"{msg.latitude:.7f},{msg.longitude:.7f} -> {lat:.7f},{lon:.7f}"
            )

        out = NavSatFix()
        out.header = msg.header
        out.status = msg.status
        out.latitude = lat
        out.longitude = lon
        out.altitude = msg.altitude
        out.position_covariance = msg.position_covariance
        out.position_covariance_type = msg.position_covariance_type
        self._pub_fix.publish(out)

        pose = PoseStamped()
        pose.header.stamp = msg.header.stamp
        pose.header.frame_id = self._path.header.frame_id
        pose.pose.position.x = east
        pose.pose.position.y = north
        pose.pose.orientation.w = 1.0
        self._path.header.stamp = msg.header.stamp
        self._path.poses.append(pose)
        if len(self._path.poses) > self._path_max:
            self._path.poses = self._path.poses[-self._path_max :]
        self._pub_path.publish(self._path)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = RobustNavSatFilter()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        # The rclpy signal handler may have shut the context down already.
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
