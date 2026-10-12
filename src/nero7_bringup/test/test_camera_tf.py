"""The base transform must attach to the camera root, preserving optical TF."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from nero7_bringup.launch_common import calibrated_camera_root_transform


def calibration():
    optical_to_base = np.eye(4)
    optical_to_base[:3, :3] = Rotation.from_euler('xyz', [.2, -.6, .8]).as_matrix()
    optical_to_base[:3, 3] = [-.8, -.16, .77]
    q = Rotation.from_euler('xyz', [-1.57, .003, -1.57]).as_quat()
    return dict(camera_frame='camera_color_optical_frame', base_frame='base_link',
                translation_unit='m', camera_to_base_cv=optical_to_base.tolist(),
                camera_link_to_color_optical=dict(translation=[0., -.024, 0.],
                                                 quaternion=q[[3, 0, 1, 2]].tolist()))


def test_root_and_driver_chain_recover_optical_calibration():
    data = calibration()
    fixed = data['camera_link_to_color_optical']
    optical_to_link = np.eye(4)
    q = np.array(fixed['quaternion'])
    optical_to_link[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    optical_to_link[:3, 3] = fixed['translation']
    assert np.allclose(calibrated_camera_root_transform(data) @ optical_to_link,
                       data['camera_to_base_cv'], atol=1e-12)


@pytest.mark.parametrize('field,value', [('translation_unit', 'mm'),
                                        ('camera_frame', 'camera_depth_optical_frame')])
def test_reject_wrong_units_or_optical_frame(field, value):
    data = calibration()
    data[field] = value
    with pytest.raises(ValueError):
        calibrated_camera_root_transform(data)


def test_missing_driver_extrinsic_cannot_fall_back_to_identity():
    data = calibration()
    del data['camera_link_to_color_optical']
    with pytest.raises(KeyError):
        calibrated_camera_root_transform(data)


def test_invalid_driver_quaternion_is_rejected():
    data = calibration()
    data['camera_link_to_color_optical']['quaternion'] = [0., 0., 0., 0.]
    with pytest.raises(ValueError):
        calibrated_camera_root_transform(data)
