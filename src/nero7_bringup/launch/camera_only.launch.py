import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    camera_name = LaunchConfiguration('camera_name')
    depth_registration = LaunchConfiguration('depth_registration')
    enable_colored_point_cloud = LaunchConfiguration('enable_colored_point_cloud')

    return LaunchDescription([
        DeclareLaunchArgument('camera_name', default_value='camera'),
        DeclareLaunchArgument('depth_registration', default_value='true'),
        DeclareLaunchArgument('enable_colored_point_cloud', default_value='true'),
        DeclareLaunchArgument('enable_color_undistortion', default_value='true',
                              description='Undistort RGB pixels to match registered depth and pinhole inference'),
        DeclareLaunchArgument('depth_width', default_value='0',
                              description='0 = device default (848x480)'),
        DeclareLaunchArgument('depth_height', default_value='0'),
        DeclareLaunchArgument('color_width', default_value='0',
                              description='0 = device default (1280x720)'),
        DeclareLaunchArgument('color_height', default_value='0'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('orbbec_camera'),
                    'launch', 'gemini_330_series.launch.py'])),
            launch_arguments={
                'camera_name': camera_name,
                'depth_registration': depth_registration,
                'enable_colored_point_cloud': enable_colored_point_cloud,
                'enable_color_undistortion': LaunchConfiguration('enable_color_undistortion'),
                'depth_width': LaunchConfiguration('depth_width'),
                'depth_height': LaunchConfiguration('depth_height'),
                'color_width': LaunchConfiguration('color_width'),
                'color_height': LaunchConfiguration('color_height'),
            }.items(),
        ),
    ])
