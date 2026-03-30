"""ROS2 launch file for the GNSS GPU positioning node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate a launch description for the GNSS GPU positioning node."""

    n_particles_arg = DeclareLaunchArgument(
        'n_particles', default_value='100000',
        description='Number of particles for the particle filter'
    )
    sigma_pr_arg = DeclareLaunchArgument(
        'sigma_pr', default_value='5.0',
        description='Pseudorange observation noise std dev [m]'
    )
    sigma_pos_arg = DeclareLaunchArgument(
        'sigma_pos', default_value='1.0',
        description='Position prediction noise std dev [m]'
    )
    resampling_arg = DeclareLaunchArgument(
        'resampling_method', default_value='megopolis',
        description='Resampling method: megopolis or systematic'
    )
    use_3d_model_arg = DeclareLaunchArgument(
        'use_3d_model', default_value='false',
        description='Whether to use 3D building model for ray tracing'
    )
    building_model_path_arg = DeclareLaunchArgument(
        'building_model_path', default_value='',
        description='Path to 3D building model file'
    )

    gnss_node = Node(
        package='gnss_gpu',
        executable='gnss_positioning_node',
        name='gnss_gpu_positioning',
        parameters=[{
            'n_particles': LaunchConfiguration('n_particles'),
            'sigma_pr': LaunchConfiguration('sigma_pr'),
            'sigma_pos': LaunchConfiguration('sigma_pos'),
            'resampling_method': LaunchConfiguration('resampling_method'),
            'use_3d_model': LaunchConfiguration('use_3d_model'),
            'building_model_path': LaunchConfiguration('building_model_path'),
        }],
        remappings=[
            ('/gnss/raw', '/ublox/fix'),
        ],
        output='screen',
    )

    return LaunchDescription([
        n_particles_arg,
        sigma_pr_arg,
        sigma_pos_arg,
        resampling_arg,
        use_3d_model_arg,
        building_model_path_arg,
        gnss_node,
    ])
