"""Full system: camera + arm + TF + perception + control."""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import PathJoinSubstitution

from nero7_bringup.launch_common import camera_static_tf_opaque


def generate_launch_description():
    this_dir = os.path.dirname(os.path.realpath(__file__))
    pkg_dir = os.path.dirname(this_dir)  # .../src/nero7_bringup

    calib_path = LaunchConfiguration('calibration_file')
    perception_params = LaunchConfiguration('perception_params')
    controller_params = LaunchConfiguration('controller_params')

    return LaunchDescription([
        DeclareLaunchArgument('mode', default_value='conveyor'),
        DeclareLaunchArgument('execution_enabled', default_value='false'),
        DeclareLaunchArgument('calibration_file',
                              default_value=os.path.join(pkg_dir, 'config', 'calibration.yaml')),
        DeclareLaunchArgument('perception_params',
                              default_value=PathJoinSubstitution([FindPackageShare('nero7_perception'), 'config', 'perception.yaml'])),
        DeclareLaunchArgument('controller_params',
                              default_value=PathJoinSubstitution([FindPackageShare('nero7_control'), 'config', 'controller.yaml'])),
        DeclareLaunchArgument('dry_run', default_value='false'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(this_dir, 'camera_only.launch.py'))),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(this_dir, 'arm_only.launch.py'))),

        OpaqueFunction(function=camera_static_tf_opaque),

        Node(
            package='nero7_perception', executable='block_detector',
            name='block_detector', output='screen',
            parameters=[{'params_file': perception_params}]),

        Node(
            package='nero7_control', executable='grasp_controller',
            name='grasp_controller', output='screen',
            parameters=[{'params_file': controller_params,
                         'mode': LaunchConfiguration('mode'),
                         'calibration_file': LaunchConfiguration('calibration_file'),
                         'execution_enabled': ParameterValue(LaunchConfiguration('execution_enabled'), value_type=bool)},
                        {'dry_run': LaunchConfiguration('dry_run')}]),
    ])
