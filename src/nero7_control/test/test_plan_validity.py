from types import SimpleNamespace as NS
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from nero7_control.plan_validity import candidate_matches, stationary_plan_current
from nero7_control.inference_geometry import optical_rays
from nero7_perception.geometry import CameraGeometry


def example():
    track = NS(id=3, stamp=10., centroid_x=.1, centroid_y=.2, centroid_z=.04)
    result = dict(status=0, target_id=3, stamp=10., yaw=.2, raw_width_mm=30.,
                  center=np.array([.1, .2, .04]))
    rotation = Rotation.identity()
    plan = dict(track_id=3, image_stamp=10., track_center=result['center'].copy(),
                network_center=result['center'].copy(), network_yaw=.2, raw_width=.03,
                seed_joints=np.zeros(7), seed_flange=np.zeros(3), seed_rotation=rotation)
    return track, result, plan, rotation


@pytest.mark.parametrize('change', [dict(target_id=4), dict(stamp=9.),
                                   dict(stamp=10.2), dict(stamp=np.nan), dict(status=1)])
def test_candidate_rejects_wrong_owner_stale_future_and_invalid(change):
    track, result, _, _ = example()
    assert not candidate_matches(dict(result, **change), track, 10.05)


@pytest.mark.parametrize('kind', ['joint', 'pose', 'rotation', 'target', 'height', 'yaw', 'width', 'age'])
def test_execution_rejects_changed_plan_context(kind):
    track, result, plan, rotation = example()
    joints, position, now = np.zeros(7), np.zeros(3), 10.05
    assert stationary_plan_current(plan, track, result, now, joints, position, rotation)
    if kind == 'joint': joints[0] = .2
    if kind == 'pose': position[0] = .02
    if kind == 'rotation': rotation = Rotation.from_euler('z', .2)
    if kind == 'target': track.id = 4
    if kind == 'height': track.centroid_z += .02
    if kind == 'yaw': result['yaw'] += .2
    if kind == 'width': result['raw_width_mm'] += 10.
    if kind == 'age':
        now = 11.
        result['stamp'] = track.stamp = now
    assert not stationary_plan_current(plan, track, result, now, joints, position, rotation)


@pytest.mark.parametrize('model,distortion', [('plumb_bob', [.2, -.03, .001, 0., 0.]),
                                            ('equidistant', [.02, .01, 0., 0.])])
def test_network_and_detector_use_same_distorted_optical_rays(model, distortion):
    K = np.array([[500., 0., 320.], [0., 510., 240.], [0., 0., 1.]])
    pixels = np.array([[100., 80.], [320., 240.], [550., 410.]])
    camera = CameraGeometry(K, 640, 480, 'optical', distortion, model)
    np.testing.assert_allclose(optical_rays(pixels, K, distortion, model),
                               camera.rays(pixels[:, 0], pixels[:, 1]), atol=1e-12)


def test_minimum_tilt_alignment_does_not_add_twist_to_horizontal_reference():
    from nero7_control.kinematics import yaw_orientations
    yaw = 1.444185508449333
    rotations = list(yaw_orientations([0., 0., 0.], [1., 0., 0.], yaw))
    expected = Rotation.from_euler('z', yaw)
    assert min((rotation.inv()*expected).magnitude() for rotation in rotations) < 1e-12
