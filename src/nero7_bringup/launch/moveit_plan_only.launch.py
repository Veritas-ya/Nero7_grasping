"""Collision-aware MoveIt planning server without a competing arm controller."""

import os
import sys
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _start(context):
    moveit_share = Path(get_package_share_directory('agx_arm_moveit'))
    sys.path.insert(0, str(moveit_share/'launch'))
    from _moveit_config_builder import build_moveit_config

    config = build_moveit_config(context)
    params = config.to_dict()
    # The Orbbec launch used by this project publishes this registered cloud.
    # Override the upstream RealSense-oriented topic and reduce the excessive
    # 100 mm obstacle padding to a still-conservative 25 mm.
    params.update({
        'sensors': ['point_cloud_sensor'],
        'octomap_frame': 'base_link',
        'octomap_resolution': .02,
        'point_cloud_sensor.sensor_plugin':
            'occupancy_map_monitor/PointCloudOctomapUpdater',
        'point_cloud_sensor.point_cloud_topic':
            LaunchConfiguration('point_cloud_topic').perform(context),
        'point_cloud_sensor.max_range': 2.0,
        'point_cloud_sensor.point_subsample': 1,
        'point_cloud_sensor.padding_offset': .025,
        'point_cloud_sensor.padding_scale': 1.0,
        'point_cloud_sensor.max_update_rate': 2.0,
        'point_cloud_sensor.filtered_cloud_topic': '/moveit/filtered_cloud',
        'publish_robot_description_semantic': True,
        'allow_trajectory_execution': False,
        'publish_planning_scene': True,
        'publish_geometry_updates': True,
        'publish_state_updates': True,
        'publish_transforms_updates': True,
        'monitor_dynamics': False,
        'planning_pipelines': ['ompl'],
        'default_planning_pipeline': 'ompl',
        'use_sim_time': LaunchConfiguration('use_sim_time').perform(context) == 'true',
    })
    feedback = LaunchConfiguration('feedback_topic').perform(context)
    return [
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             output='screen', parameters=[config.robot_description],
             remappings=[('joint_states', feedback)]),
        Node(package='moveit_ros_move_group', executable='move_group',
             output='screen', parameters=[params],
             remappings=[('joint_states', feedback)],
             additional_env={'DISPLAY': os.environ.get('DISPLAY', '')}),
    ]


def generate_launch_description():
    # These names match agx_arm_moveit's builder contract.
    return LaunchDescription([
        DeclareLaunchArgument('namespace', default_value=''),
        DeclareLaunchArgument('arm_type', default_value='nero'),
        DeclareLaunchArgument('effector_type', default_value='agx_gripper'),
        DeclareLaunchArgument('revo2_type', default_value='left'),
        DeclareLaunchArgument('tcp_offset', default_value=
                              '[0.1831669048, 0.0076336831, -0.0256367994, 0.0, 0.0, 0.0]'),
        DeclareLaunchArgument('follow', default_value='true'),
        DeclareLaunchArgument('feedback_topic', default_value='feedback/joint_states'),
        DeclareLaunchArgument('control_topic', default_value='control/joint_states'),
        DeclareLaunchArgument('point_cloud_topic', default_value='/camera/depth/points'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        OpaqueFunction(function=_start),
    ])
