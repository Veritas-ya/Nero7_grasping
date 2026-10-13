"""Early trajectory/following preview; optional execution with an existing driver.

Start the camera/TF/detector through perception_preview. The AGX arm driver must
already be running (one CAN owner). execution_enabled defaults to false.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    share = FindPackageShare('nero7_bringup')
    return LaunchDescription([
        DeclareLaunchArgument('execution_enabled', default_value='false'),
        DeclareLaunchArgument('start_camera', default_value='false'),
        DeclareLaunchArgument('publish_camera_tf', default_value='true'),
        DeclareLaunchArgument('detection_mode', default_value='legacy_depth'),
        DeclareLaunchArgument('visualize', default_value='true'),
        DeclareLaunchArgument('plane_mode', default_value='fixed'),
        DeclareLaunchArgument('plane_file', default_value='/tmp/nero7_belt_plane.yaml'),
        DeclareLaunchArgument('roi_px', default_value='[0,0,0,0]'),
        DeclareLaunchArgument('plane_roi_px', default_value='[0,0,0,0]'),
        DeclareLaunchArgument('perception_params', default_value=PathJoinSubstitution(
            [FindPackageShare('nero7_perception'), 'config', 'perception.yaml'])),
        DeclareLaunchArgument('calibration_file', default_value=PathJoinSubstitution(
            [share, 'config', 'calibration.yaml'])),
        DeclareLaunchArgument('controller_params', default_value=PathJoinSubstitution(
            [FindPackageShare('nero7_control'), 'config', 'follow.yaml'])),
        DeclareLaunchArgument('urdf_path', default_value=
            '/home/ubuntu22/nero_ws/install/nero_description/share/nero_description/urdf/nero_description.urdf'),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(PathJoinSubstitution(
            [share, 'launch', 'perception_preview.launch.py'])),
            launch_arguments={name: LaunchConfiguration(name) for name in (
                'start_camera', 'publish_camera_tf', 'detection_mode', 'calibration_file',
                'visualize', 'plane_mode', 'plane_file', 'roi_px', 'plane_roi_px', 'perception_params')}.items()),
        Node(package='nero7_control', executable='grasp_controller', name='grasp_controller',
            output='screen', parameters=[{
                'params_file': LaunchConfiguration('controller_params'),
                'mode': 'conveyor_follow',
                'calibration_file': LaunchConfiguration('calibration_file'),
                'urdf_path': LaunchConfiguration('urdf_path'),
                'execution_enabled': ParameterValue(LaunchConfiguration('execution_enabled'), value_type=bool)}]),
    ])
