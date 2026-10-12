"""Arm only: agx_arm_ctrl driver + robot_state_publisher for the Nero7.

Prereqs:
    - CAN up: bash ~/agx_arm_ws/src/agx_arm_ros/scripts/can_activate.sh can0 1000000
    - source ~/agx_arm_ws/install/setup.bash (for agx_arm_ctrl / agx_arm_msgs)

robot_state_publisher is fed with /feedback/joint_states (remapped) so TF
follows the real arm.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

from nero7_bringup.launch_common import DEFAULT_URDF


def generate_launch_description():
    can_port = LaunchConfiguration('can_port')
    arm_type = LaunchConfiguration('arm_type')
    effector_type = LaunchConfiguration('effector_type')
    auto_enable = LaunchConfiguration('auto_enable')
    speed_percent = LaunchConfiguration('speed_percent')
    urdf_path = LaunchConfiguration('urdf_path')

    return LaunchDescription([
        DeclareLaunchArgument('can_port', default_value='can0'),
        DeclareLaunchArgument('arm_type', default_value='nero'),
        DeclareLaunchArgument('effector_type', default_value='agx_gripper'),
        DeclareLaunchArgument('auto_enable', default_value='true'),
        DeclareLaunchArgument('speed_percent', default_value='50',
                              description='arm speed limit in percent'),
        DeclareLaunchArgument('urdf_path', default_value=DEFAULT_URDF),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('agx_arm_ctrl'),
                    'launch', 'start_single_agx_arm.launch.py'])),
            launch_arguments={
                'can_port': can_port,
                'arm_type': arm_type,
                'effector_type': effector_type,
                'auto_enable': auto_enable,
                'speed_percent': speed_percent,
            }.items(),
        ),

        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{'robot_description': ParameterValue(
                Command(['cat ', urdf_path]), value_type=str)}],
            remappings=[('/joint_states', '/feedback/joint_states')],
        ),
    ])
