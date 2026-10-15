"""Shared launch helpers for the Nero7 grasping bringup."""
import yaml
import numpy as np
from scipy.spatial.transform import Rotation

PKG = 'nero7_bringup'
DEFAULT_URDF = ('/home/ubuntu22/nero_ws/install/nero_description/share/'
                'nero_description/urdf/nero_description.urdf')


def calibrated_camera_root_transform(data):
    """Strict base -> camera_link composition; preserve the driver's optical TF."""
    from nero7_perception.geometry import check_transform
    if (data.get('camera_frame') != 'camera_color_optical_frame'
            or data.get('base_frame') != 'base_link' or data.get('translation_unit') != 'm'):
        raise ValueError('Calibration must describe color optical -> base_link in metres')
    optical_to_base = check_transform(data['camera_to_base_cv'])
    fixed = data['camera_link_to_color_optical']
    translation = np.asarray(fixed['translation'], float)
    quaternion = np.asarray(fixed['quaternion'], float)  # wxyz
    if (translation.shape != (3,) or not np.isfinite(translation).all()
            or quaternion.shape != (4,) or not np.isfinite(quaternion).all()
            or not np.isclose(np.linalg.norm(quaternion), 1., atol=1e-5)):
        raise ValueError('Invalid camera_link -> color optical calibration')
    optical_to_link = np.eye(4)
    optical_to_link[:3, :3] = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
    optical_to_link[:3, 3] = translation
    return check_transform(optical_to_base @ np.linalg.inv(optical_to_link))


def load_calibration(path):
    """Load a validated optical calibration; missing data must fail startup."""
    with open(path) as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError('Calibration must be a YAML mapping')
    transform = calibrated_camera_root_transform(data)
    quaternion = Rotation.from_matrix(transform[:3, :3]).as_quat()
    return [*transform[:3, 3].tolist(), *quaternion.tolist()]


def camera_static_tf_opaque(context):
    """Publish the calibrated base_link -> camera_link transform."""
    from launch.substitutions import LaunchConfiguration
    from launch_ros.actions import Node
    path = LaunchConfiguration('calibration_file').perform(context)
    args = [str(v) for v in load_calibration(path)]
    args += ['base_link', 'camera_link']
    return [Node(package='tf2_ros', executable='static_transform_publisher',
                 arguments=args)]
