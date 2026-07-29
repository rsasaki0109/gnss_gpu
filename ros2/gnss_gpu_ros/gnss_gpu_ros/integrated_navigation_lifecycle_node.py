"""ROS 2 lifecycle wrapper for deterministic GNSS/IMU/map safety handling."""

from __future__ import annotations

import json
from typing import Any

from gnss_gpu_ros.lifecycle_core import (
    EventDisposition,
    LifecycleParameters,
    NavigationLifecycleCore,
    NavigationMode,
    SensorEvent,
)

try:
    import rclpy
    from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
    from rcl_interfaces.msg import SetParametersResult
    from rclpy.executors import ExternalShutdownException
    from rclpy.lifecycle import LifecycleNode, State, TransitionCallbackReturn
    from sensor_msgs.msg import Imu, NavSatFix
    from std_msgs.msg import String

    HAS_ROS2 = True
except ImportError:  # pragma: no cover - exercised on a ROS installation
    HAS_ROS2 = False
    LifecycleNode = object  # type: ignore[assignment,misc]


_CORE_PARAMETER_NAMES = tuple(LifecycleParameters.__dataclass_fields__)


def _seconds(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _fix_payload(message: Any) -> dict[str, Any]:
    return {
        "latitude": float(message.latitude),
        "longitude": float(message.longitude),
        "altitude": float(message.altitude),
        "status": int(message.status.status),
        "service": int(message.status.service),
        "covariance": [float(item) for item in message.position_covariance],
        "covariance_type": int(message.position_covariance_type),
        "frame_id": str(message.header.frame_id),
    }


def _imu_payload(message: Any) -> dict[str, Any]:
    return {
        "orientation": [
            float(message.orientation.x),
            float(message.orientation.y),
            float(message.orientation.z),
            float(message.orientation.w),
        ],
        "angular_velocity": [
            float(message.angular_velocity.x),
            float(message.angular_velocity.y),
            float(message.angular_velocity.z),
        ],
        "linear_acceleration": [
            float(message.linear_acceleration.x),
            float(message.linear_acceleration.y),
            float(message.linear_acceleration.z),
        ],
        "frame_id": str(message.header.frame_id),
    }


if HAS_ROS2:

    class IntegratedNavigationLifecycleNode(LifecycleNode):
        """Managed node with fail-closed timestamp and watchdog behavior."""

        def __init__(self) -> None:
            super().__init__("gnss_gpu_navigation")
            defaults = LifecycleParameters()
            for name in _CORE_PARAMETER_NAMES:
                self.declare_parameter(name, getattr(defaults, name))
            self.declare_parameter("gnss_topic", "fix")
            self.declare_parameter("imu_topic", "imu/data")
            self.declare_parameter("map_topic", "map_context")
            self.declare_parameter("output_topic", "navigation/fix")
            self.declare_parameter("diagnostics_topic", "navigation/diagnostics")

            self._core = NavigationLifecycleCore()
            self._fix_subscription = None
            self._imu_subscription = None
            self._map_subscription = None
            self._fix_publisher = None
            self._diagnostics_publisher = None
            self._watchdog_timer = None
            self._parameter_callback = self.add_on_set_parameters_callback(
                self._validate_parameter_update
            )

        def _core_parameter_values(self) -> dict[str, Any]:
            return {
                name: self.get_parameter(name).value
                for name in _CORE_PARAMETER_NAMES
            }

        def _validate_parameter_update(self, parameters) -> SetParametersResult:
            if self._core.state.value == "active":
                return SetParametersResult(
                    successful=False,
                    reason="deactivate before changing navigation parameters",
                )
            values = self._core_parameter_values()
            for parameter in parameters:
                if parameter.name in _CORE_PARAMETER_NAMES:
                    values[parameter.name] = parameter.value
                elif parameter.name.endswith("_topic"):
                    if not isinstance(parameter.value, str) or not parameter.value.strip():
                        return SetParametersResult(
                            successful=False,
                            reason=f"{parameter.name} must be a non-empty string",
                        )
            try:
                LifecycleParameters.from_mapping(values)
            except ValueError as exc:
                return SetParametersResult(successful=False, reason=str(exc))
            return SetParametersResult(successful=True)

        def on_configure(self, state: State) -> TransitionCallbackReturn:
            del state
            try:
                self._core.configure(self._core_parameter_values())
                topics = {
                    name: str(self.get_parameter(name).value)
                    for name in (
                        "gnss_topic",
                        "imu_topic",
                        "map_topic",
                        "output_topic",
                        "diagnostics_topic",
                    )
                }
                if not all(value.strip() for value in topics.values()):
                    raise ValueError("topic parameters must be non-empty")
                self._fix_publisher = self.create_lifecycle_publisher(
                    NavSatFix,
                    topics["output_topic"],
                    10,
                )
                self._diagnostics_publisher = self.create_lifecycle_publisher(
                    DiagnosticArray,
                    topics["diagnostics_topic"],
                    10,
                )
                self._fix_subscription = self.create_subscription(
                    NavSatFix,
                    topics["gnss_topic"],
                    self._on_fix,
                    50,
                )
                self._imu_subscription = self.create_subscription(
                    Imu,
                    topics["imu_topic"],
                    self._on_imu,
                    200,
                )
                self._map_subscription = self.create_subscription(
                    String,
                    topics["map_topic"],
                    self._on_map,
                    10,
                )
                period = float(self.get_parameter("diagnostics_period_s").value)
                self._watchdog_timer = self.create_timer(period, self._on_watchdog)
            except (RuntimeError, ValueError) as exc:
                self.get_logger().error(f"configuration rejected: {exc}")
                return TransitionCallbackReturn.FAILURE
            self.get_logger().info(
                "configured GNSS/IMU/map inputs, navigation fix, and diagnostics"
            )
            return TransitionCallbackReturn.SUCCESS

        def on_activate(self, state: State) -> TransitionCallbackReturn:
            del state
            try:
                self._core.activate()
                self._fix_publisher.on_activate()
                self._diagnostics_publisher.on_activate()
            except RuntimeError as exc:
                self.get_logger().error(str(exc))
                return TransitionCallbackReturn.FAILURE
            self.get_logger().info("navigation lifecycle node active")
            return TransitionCallbackReturn.SUCCESS

        def on_deactivate(self, state: State) -> TransitionCallbackReturn:
            del state
            try:
                self._core.deactivate()
                self._fix_publisher.on_deactivate()
                self._diagnostics_publisher.on_deactivate()
            except RuntimeError as exc:
                self.get_logger().error(str(exc))
                return TransitionCallbackReturn.FAILURE
            return TransitionCallbackReturn.SUCCESS

        def on_cleanup(self, state: State) -> TransitionCallbackReturn:
            del state
            try:
                self._core.cleanup()
            except RuntimeError as exc:
                self.get_logger().error(str(exc))
                return TransitionCallbackReturn.FAILURE
            for subscription in (
                self._fix_subscription,
                self._imu_subscription,
                self._map_subscription,
            ):
                if subscription is not None:
                    self.destroy_subscription(subscription)
            if self._watchdog_timer is not None:
                self.destroy_timer(self._watchdog_timer)
            for publisher in (
                self._fix_publisher,
                self._diagnostics_publisher,
            ):
                if publisher is not None:
                    self.destroy_publisher(publisher)
            self._fix_subscription = None
            self._imu_subscription = None
            self._map_subscription = None
            self._watchdog_timer = None
            self._fix_publisher = None
            self._diagnostics_publisher = None
            return TransitionCallbackReturn.SUCCESS

        def on_shutdown(self, state: State) -> TransitionCallbackReturn:
            del state
            self._core.shutdown()
            return TransitionCallbackReturn.SUCCESS

        def on_error(self, state: State) -> TransitionCallbackReturn:
            del state
            self._core.fail("ros_lifecycle_error")
            self.get_logger().error("lifecycle error; navigation forced to safe fallback")
            return TransitionCallbackReturn.SUCCESS

        def _event_time(self, message: Any) -> tuple[float, float]:
            arrival = self.get_clock().now().nanoseconds * 1e-9
            stamp = _seconds(message.header.stamp)
            return (arrival if stamp == 0.0 else stamp), arrival

        def _on_fix(self, message: NavSatFix) -> None:
            stamp, arrival = self._event_time(message)
            result = self._core.ingest(
                SensorEvent.create("gnss", stamp, arrival, _fix_payload(message))
            )
            self._log_rejection(result.disposition, result.reason)
            if result.output is None or self._fix_publisher is None:
                return
            output = NavSatFix()
            output.header = message.header
            if _seconds(output.header.stamp) == 0.0:
                output.header.stamp = self.get_clock().now().to_msg()
            output.status = message.status
            output.latitude = float(result.output["latitude"])
            output.longitude = float(result.output["longitude"])
            output.altitude = float(result.output["altitude"])
            output.position_covariance = list(message.position_covariance)
            if result.mode != NavigationMode.NORMAL:
                floor = float(result.output["covariance_floor_m2"])
                for index in (0, 4, 8):
                    output.position_covariance[index] = max(
                        float(output.position_covariance[index]),
                        floor,
                    )
                output.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
            else:
                output.position_covariance_type = message.position_covariance_type
            self._fix_publisher.publish(output)

        def _on_imu(self, message: Imu) -> None:
            stamp, arrival = self._event_time(message)
            result = self._core.ingest(
                SensorEvent.create("imu", stamp, arrival, _imu_payload(message))
            )
            self._log_rejection(result.disposition, result.reason)

        def _on_map(self, message: String) -> None:
            arrival = self.get_clock().now().nanoseconds * 1e-9
            try:
                decoded = json.loads(message.data)
                payload = decoded if isinstance(decoded, dict) else {"map_id": message.data}
            except json.JSONDecodeError:
                payload = {"map_id": message.data}
            result = self._core.ingest(
                SensorEvent.create("map", arrival, arrival, payload)
            )
            self._log_rejection(result.disposition, result.reason)

        def _log_rejection(
            self,
            disposition: EventDisposition,
            reason: str,
        ) -> None:
            if disposition not in {
                EventDisposition.ACCEPTED,
                EventDisposition.DUPLICATE,
            }:
                self.get_logger().warn(
                    f"input {disposition.value}: {reason}"
                )

        def _on_watchdog(self) -> None:
            now = self.get_clock().now()
            snapshot = self._core.watchdog(now.nanoseconds * 1e-9)
            if self._diagnostics_publisher is None:
                return
            array = DiagnosticArray()
            array.header.stamp = now.to_msg()
            status = DiagnosticStatus()
            status.name = f"{self.get_name()}:runtime"
            status.hardware_id = "gnss_gpu"
            if snapshot.navigation_mode == NavigationMode.NORMAL:
                status.level = DiagnosticStatus.OK
            else:
                status.level = DiagnosticStatus.WARN
            status.message = snapshot.reason
            values = {
                "lifecycle_state": snapshot.lifecycle_state.value,
                "navigation_mode": snapshot.navigation_mode.value,
                **{
                    f"{sensor}_age_s": "missing" if age is None else f"{age:.6f}"
                    for sensor, age in snapshot.sensor_age_s.items()
                },
                **{
                    f"count_{name}": str(count)
                    for name, count in snapshot.counters.items()
                },
            }
            status.values = [
                KeyValue(key=key, value=value) for key, value in values.items()
            ]
            array.status = [status]
            self._diagnostics_publisher.publish(array)


else:

    class IntegratedNavigationLifecycleNode:  # pragma: no cover
        def __init__(self) -> None:
            raise RuntimeError("ROS 2 rclpy and message packages are required")


def main(args: list[str] | None = None) -> None:
    if not HAS_ROS2:
        raise RuntimeError("ROS 2 rclpy and message packages are required")
    rclpy.init(args=args)
    node = IntegratedNavigationLifecycleNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
