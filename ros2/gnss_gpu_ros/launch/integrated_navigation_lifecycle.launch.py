from launch import LaunchDescription
from launch.actions import EmitEvent, RegisterEventHandler
from launch.events.matchers import matches_action
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from launch_ros.actions import LifecycleNode
from lifecycle_msgs.msg import Transition


def generate_launch_description() -> LaunchDescription:
    node = LifecycleNode(
        package="gnss_gpu_ros",
        executable="integrated_navigation_lifecycle",
        name="gnss_gpu_navigation",
        namespace="",
        output="screen",
        parameters=[
            {
                "gnss_topic": "fix",
                "imu_topic": "imu/data",
                "map_topic": "map_context",
                "output_topic": "navigation/fix",
                "diagnostics_topic": "navigation/diagnostics",
                "gnss_timeout_s": 1.5,
                "imu_timeout_s": 0.25,
                "map_timeout_s": 10.0,
                "maximum_future_skew_s": 0.2,
                "require_imu": True,
                "require_map": True,
                "reject_out_of_order": True,
                "fallback_covariance_m2": 10000.0,
                "diagnostics_period_s": 0.5,
            }
        ],
    )
    return LaunchDescription(
        [
            node,
            RegisterEventHandler(
                OnStateTransition(
                    target_lifecycle_node=node,
                    goal_state="inactive",
                    entities=[
                        EmitEvent(
                            event=ChangeState(
                                lifecycle_node_matcher=matches_action(node),
                                transition_id=Transition.TRANSITION_ACTIVATE,
                            )
                        )
                    ],
                )
            ),
            EmitEvent(
                event=ChangeState(
                    lifecycle_node_matcher=matches_action(node),
                    transition_id=Transition.TRANSITION_CONFIGURE,
                )
            )
        ]
    )
