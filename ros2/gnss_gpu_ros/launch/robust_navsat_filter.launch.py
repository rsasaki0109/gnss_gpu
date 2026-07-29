from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            Node(
                package="gnss_gpu_ros",
                executable="robust_navsat_filter",
                name="robust_navsat_filter",
                output="screen",
                parameters=[
                    {
                        "hampel_window": 21,
                        "hampel_k": 2.5,
                        "kalman_sigma_a": 1.0,
                        "kalman_sigma_z": 1.0,
                        "use_hampel": True,
                "use_kalman": True,
                "max_gap_s": 30.0,
                        "path_frame_id": "map",
                    }
                ],
                # remappings=[("fix", "/your_gnss_driver/fix")],
            ),
        ]
    )
