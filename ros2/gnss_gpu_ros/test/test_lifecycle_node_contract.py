from __future__ import annotations

from types import SimpleNamespace

import pytest

from gnss_gpu_ros.integrated_navigation_lifecycle_node import (
    HAS_ROS2,
    IntegratedNavigationLifecycleNode,
    _fix_payload,
    _imu_payload,
)


def vector(x, y, z):
    return SimpleNamespace(x=x, y=y, z=z)


def test_message_payload_helpers_are_deterministic() -> None:
    fix = SimpleNamespace(
        latitude=35.0,
        longitude=139.0,
        altitude=40.0,
        status=SimpleNamespace(status=1, service=3),
        position_covariance=[1.0] * 9,
        position_covariance_type=2,
        header=SimpleNamespace(frame_id="gnss"),
    )
    assert _fix_payload(fix) == _fix_payload(fix)
    imu = SimpleNamespace(
        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        angular_velocity=vector(0.1, 0.2, 0.3),
        linear_acceleration=vector(1.0, 2.0, 3.0),
        header=SimpleNamespace(frame_id="imu"),
    )
    assert _imu_payload(imu)["linear_acceleration"] == [1.0, 2.0, 3.0]


def test_missing_ros_runtime_fails_explicitly() -> None:
    if HAS_ROS2:
        pytest.skip("ROS 2 is installed")
    with pytest.raises(RuntimeError, match="ROS 2"):
        IntegratedNavigationLifecycleNode()
