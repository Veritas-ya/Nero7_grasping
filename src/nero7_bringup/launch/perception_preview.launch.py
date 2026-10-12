"""Perception/empty-belt calibration only. No arm driver or grasp controller."""
import json
import os
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from scipy.spatial.transform import Rotation


def start(context):
    arg = lambda name: LaunchConfiguration(name).perform(context)
    share = Path(get_package_share_directory('nero7_bringup'))
    actions = []
    if arg('start_camera') == 'true':
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(share/'launch/camera_only.launch.py')),
            launch_arguments={'depth_registration': 'true'}.items()))
    if arg('publish_camera_tf') == 'true':
        # This optional publisher is strict; missing/invalid calibration cannot become identity TF.
        import numpy as np
        from nero7_bringup.launch_common import calibrated_camera_root_transform
        calibration = yaml.safe_load(Path(arg('calibration_file')).read_text())
        transform = calibrated_camera_root_transform(calibration)
        quaternion = Rotation.from_matrix(transform[:3, :3]).as_quat()
        actions.append(Node(package='tf2_ros', executable='static_transform_publisher',
            arguments=[str(v) for v in np.r_[transform[:3, 3], quaternion]]+
                       ['base_link', 'camera_link']))
    roi = json.loads(arg('roi_px'))
    plane_roi = json.loads(arg('plane_roi_px'))
    if any(len(value) != 4 or any(type(v) is not int for v in value) for value in (roi, plane_roi)):
        raise ValueError('ROIs must be JSON lists of four integer pixel coordinates')
    actions.append(Node(package='nero7_perception', executable='block_detector', output='screen',
        parameters=[dict(params_file=arg('perception_params'), detection_mode=arg('detection_mode'),
            plane_mode=arg('plane_mode'), plane_file=arg('plane_file'),
            plane_calibration_enabled=arg('empty_belt_calibration') == 'true',
            roi_px=roi, plane_roi_px=plane_roi, calibration_file=arg('calibration_file'))]))
    if arg('visualize') == 'true':
        if os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'):
            params = yaml.safe_load(Path(arg('perception_params')).read_text())['block_detector']['ros__parameters']
            actions.append(Node(package='rqt_image_view', executable='rqt_image_view',
                                arguments=[params.get('debug_topic', '/perception/debug_image')]))
        else:
            actions.append(LogInfo(msg='No desktop display; inspect /perception/diagnostics and debug image topic.'))
    return actions


def generate_launch_description():
    bringup = Path(get_package_share_directory('nero7_bringup'))
    perception = Path(get_package_share_directory('nero7_perception'))
    defaults = dict(start_camera='false', publish_camera_tf='false', visualize='true',
        detection_mode='rgb_depth', plane_mode='fixed', plane_file='/tmp/nero7_belt_plane.yaml',
        empty_belt_calibration='false', roi_px='[0,0,0,0]', plane_roi_px='[0,0,0,0]',
        perception_params=str(perception/'config/perception.yaml'),
        calibration_file=str(bringup/'config/calibration.yaml'))
    return LaunchDescription([DeclareLaunchArgument(key, default_value=value) for key, value in defaults.items()]
                             + [OpaqueFunction(function=start)])
