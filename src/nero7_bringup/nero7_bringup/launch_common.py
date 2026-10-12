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
    """Return base->camera_link, converting optical calibration when specified."""
    t = [0.0, 0.0, 0.0]
    q = [1.0, 0.0, 0.0, 0.0]  # w, x, y, z
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
        if data and 'camera_in_base' in data:
            t = data['camera_in_base']['translation']
            q = data['camera_in_base']['quaternion']
            if data.get('camera_frame') == 'camera_color_optical_frame':
                # T_base_link = T_base_optical @ inverse(T_link_optical).
                fixed = data['camera_link_to_color_optical']
                fq = fixed['quaternion']  # wxyz
                base_optical = Rotation.from_quat([q[1], q[2], q[3], q[0]])
                link_optical = Rotation.from_quat([fq[1], fq[2], fq[3], fq[0]])
                base_link = base_optical * link_optical.inv()
                t = np.asarray(t) - base_link.apply(fixed['translation'])
                x, y, z, w = base_link.as_quat()
                q = [w, x, y, z]
    except (OSError, KeyError, TypeError):
        pass
    return [t[0], t[1], t[2], q[1], q[2], q[3], q[0]]


def camera_static_tf_opaque(context):
    """Publish the calibrated base_link -> camera_link transform."""
    from launch.substitutions import LaunchConfiguration
    from launch_ros.actions import Node
    path = LaunchConfiguration('calibration_file').perform(context)
    args = [str(v) for v in load_calibration(path)]
    args += ['base_link', 'camera_link']
    return [Node(package='tf2_ros', executable='static_transform_publisher',
                 arguments=args)]
