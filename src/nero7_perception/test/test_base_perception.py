"""Geometry, robust empty-belt calibration, detection and tracking contracts."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
from builtin_interfaces.msg import Time
from sensor_msgs.msg import PointCloud2, PointField

from nero7_perception.geometry import CameraGeometry, depth_metres, roi_mask
from nero7_perception.base_plane import (BaseBeltPlane, EmptyBeltCalibration, fit_ransac3d,
    save_plane, load_plane, check_plane_context, plane_context)
from nero7_perception.detection import (load_detection_params, calibration_points, detect_blocks)
from nero7_perception.alignment import DepthSource, project_cloud
from nero7_perception.pcl_utils import read_points_xyz
from nero7_perception.block_detector import BlockDetectorNode
from nero7_perception.validation import synthetic_dataset, calibrate_dataset, evaluate_dataset, dataset_camera


CONFIG = Path(__file__).resolve().parents[1] / 'config/perception.yaml'


@pytest.fixture
def scenario():
    data, truth = synthetic_dataset(frames=36)
    params = load_detection_params(CONFIG)
    params.update(workspace_x=[-1., 1.], plane_roi_px=[20, 20, 300, 220], detection_mode='rgb_depth')
    camera = dataset_camera(data)
    plane, _ = calibrate_dataset(data, camera, params, 'fixed')
    return data, truth, params, camera, plane


def test_depth_units_and_all_invalid_values():
    depth = depth_metres(np.array([[1000, 0]], np.uint16), '16UC1')
    assert depth[0, 0] == 1 and np.isnan(depth[0, 1])
    values = depth_metres(np.array([[1., 0., -1., np.nan, np.inf]], np.float32), '32FC1')
    assert values[0, 0] == 1 and np.isnan(values[0, 1:]).all()
    with pytest.raises(ValueError):
        depth_metres(np.ones((2, 2), np.uint16), 'mono16')


def test_distorted_pixel_roundtrip_and_grid_rejection():
    camera = CameraGeometry([[500, 0, 320], [0, 500, 240], [0, 0, 1]], 640, 480, 'optical',
                            (.08, -.03, .001, -.001, .005))
    pixels = np.array([[20., 30.], [600., 450.], [320., 240.]])
    np.testing.assert_allclose(camera.project(camera.rays(*pixels.T)), pixels, atol=1e-3)
    with pytest.raises(ValueError, match='no resize'):
        camera.check_grid(np.ones((240, 320)), 'optical')
    with pytest.raises(ValueError):
        camera.check_grid(np.ones((480, 640)), 'depth_optical')
    with pytest.raises(ValueError):
        roi_mask((480, 640), [0, 0, 700, 480])


def test_true_xyz_ransac_rejects_outliers_and_refines_normal():
    rng = np.random.default_rng(312)
    xy = rng.uniform([-.5, -.3], [.5, .3], (1500, 2))
    z = -.1 + .035*xy[:, 0] - .012*xy[:, 1] + rng.normal(0, .0006, len(xy))
    points = np.column_stack([xy, z])
    points[:300, 2] += .05
    plane, _ = fit_ransac3d(points, min_ratio=.7)
    assert plane.diagnostics['inlier_ratio'] > .79
    np.testing.assert_allclose(plane.z_at([-.3, .3], [0., 0.]), [-.1105, -.0895], atol=.0002)
    assert plane.diagnostics['residual_p95_m'] < .002
    point = np.array([.2, .1, plane.z_at(.2, .1)+.04])
    assert plane.vertical_height(point) == pytest.approx(.04)
    assert plane.signed_distance(point) == pytest.approx(.04*plane.normal[2])


def test_degenerate_points_are_not_a_plane():
    points = np.column_stack([np.linspace(0, 1, 200), np.zeros(200), np.ones(200)])
    with pytest.raises(ValueError):
        fit_ransac3d(points)


def test_fixed_model_height_is_stable_in_oblique_camera(scenario):
    _, _, _, _, plane = scenario
    assert -plane.offset == pytest.approx(-.1028662282, abs=.0002)
    assert plane.diagnostics['tilt_rad'] < .001
    assert plane.diagnostics['frame_height_spread_m'] < .001
    assert plane.diagnostics['fixed_height_p95_m'] < .002


def test_tilt_is_rejected_by_fixed_mode_but_supported_by_3d():
    data, _ = synthetic_dataset(frames=18, tilt=.035)
    camera = dataset_camera(data)
    params = load_detection_params(CONFIG)
    params.update(plane_roi_px=[20, 20, 300, 220], plane_height_hint_enabled=False)
    with pytest.raises(ValueError, match='Horizontal assumption failed'):
        calibrate_dataset(data, camera, params, 'fixed')
    plane, _ = calibrate_dataset(data, camera, params, 'ransac3d')
    assert plane.diagnostics['tilt_rad'] == pytest.approx(.035, abs=.001)


def test_duplicate_calibration_frame_and_unstable_height_are_rejected():
    rng = np.random.default_rng(4)
    xy = rng.uniform(-.2, .2, (500, 2))
    collector = EmptyBeltCalibration(frames=3, max_height_spread=.001)
    for index, z in enumerate([-.1, -.098, -.096]):
        points = np.column_stack([xy, z+np.zeros(500)])
        collector.add(points, index)
        assert not collector.add(points, index)
    assert len(collector.samples) == 3
    with pytest.raises(ValueError, match='unstable'):
        collector.fit()


def test_gray_objects_and_unspecified_roi_cannot_calibrate(scenario):
    data, _, params, camera, _ = scenario
    with pytest.raises(ValueError, match='Gray object'):
        calibration_points(data['color_bgr'][20], data['depth_m'][20], camera, data['transforms'][20], params)
    with pytest.raises(ValueError, match='explicit'):
        calibration_points(data['color_bgr'][0], data['depth_m'][0], camera,
                           data['transforms'][0], dict(params, plane_roi_px=[0, 0, 0, 0]))


def test_saved_plane_rejects_changed_tf_intrinsics_or_offsets(scenario, tmp_path):
    data, _, params, camera, plane = scenario
    path = tmp_path/'plane.yaml'
    save_plane(path, plane, camera, data['transforms'][0], 'base_link', params['plane_roi_px'])
    restored, context = load_plane(path)
    np.testing.assert_allclose(restored.normal, plane.normal)
    check_plane_context(context, camera, data['transforms'][0], 'base_link')
    changed = data['transforms'][0].copy(); changed[0, 3] += .01
    with pytest.raises(ValueError, match='context changed'):
        check_plane_context(context, camera, changed, 'base_link')
    with pytest.raises(ValueError):
        check_plane_context(context, camera, data['transforms'][0], 'base_link', (-4, 2))


@pytest.mark.parametrize('mode', ['depth_plane', 'rgb', 'rgb_depth'])
def test_oblique_camera_localization_and_vertical_height_contract(scenario, mode):
    data, truth, params, camera, plane = scenario
    index = 24
    detections, _ = detect_blocks(data['color_bgr'][index], data['depth_m'][index], camera,
                                 data['transforms'][index], plane, dict(params, detection_mode=mode))
    assert len(detections) == 1
    detection = detections[0]
    assert set(detection) == {'u', 'v', 'centroid_base', 'height', 'width_px', 'area_px', 'belt_z'}
    assert np.linalg.norm(detection['centroid_base'] - truth[index][0]['centroid_base']) < .003
    assert detection['height'] == pytest.approx(.04, abs=.001)
    assert detection['height'] + detection['belt_z'] == pytest.approx(detection['centroid_base'][2])


def test_rgb_needs_valid_interior_depth_and_geometry_rejects_belt_patch(scenario):
    data, _, params, camera, plane = scenario
    depth = data['depth_m'][24].copy()
    gray = data['color_bgr'][24][..., 0] == 145
    depth[gray] = np.nan
    with pytest.raises(ValueError, match='insufficient interior depth'):
        detect_blocks(data['color_bgr'][24], depth, camera, data['transforms'][24], plane,
                      dict(params, detection_mode='rgb'))
    # A flat gray mark on the belt is not a block in the fused mode.
    color = data['color_bgr'][0].copy(); color[90:140, 130:190] = 145
    detections, _ = detect_blocks(color, data['depth_m'][0], camera, data['transforms'][0], plane, params)
    assert detections == []


@pytest.mark.parametrize('bigendian,datatype', [(False, 7), (True, 7), (False, 8), (True, 8)])
def test_organized_cloud_float_fields_padding_and_endianness(bigendian, datatype):
    dtype = np.dtype('f4' if datatype == 7 else 'f8').newbyteorder('>' if bigendian else '<')
    size = dtype.itemsize
    cloud = PointCloud2(height=2, width=2, point_step=3*size, row_step=6*size+8, is_bigendian=bigendian)
    cloud.fields = [PointField(name=name, offset=i*size, datatype=datatype, count=1) for i, name in enumerate('xyz')]
    buffer = bytearray(cloud.row_step*cloud.height)
    expected = np.arange(12).reshape(4, 3).astype(float)
    for i in range(3):
        view = np.ndarray((2, 2), dtype=dtype, buffer=buffer, offset=i*size,
                          strides=(cloud.row_step, cloud.point_step))
        view[:] = expected[:, i].reshape(2, 2)
    cloud.data = bytes(buffer)
    np.testing.assert_array_equal(read_points_xyz(cloud), expected)


def test_projection_selects_nearest_z_not_input_order():
    camera = CameraGeometry([[100, 0, 10], [0, 100, 10], [0, 0, 1]], 20, 20, 'color')
    points = np.array([[0, 0, 2], [0, 0, .5], [0, 0, 1], [np.nan, 0, 1], [0, 0, -1]])
    depth = project_cloud(points, np.eye(4), camera)
    assert depth[10, 10] == .5
    assert np.isfinite(depth).sum() == 1


def test_registered_depth_buffer_pairs_once_and_checks_camera():
    source = DepthSource(Mock())
    camera = CameraGeometry([[100, 0, 10], [0, 100, 10], [0, 0, 1]], 20, 20, 'color')
    for nanosec in [0, 33_000_000, 66_000_000]:
        source._append(np.ones((20, 20)), Time(sec=1, nanosec=nanosec), 'color', camera)
    assert source.closest(Time(sec=1, nanosec=35_000_000), .01).stamp.nanosec == 33_000_000
    assert source.closest(Time(sec=1, nanosec=35_000_000), .01, after=1.033) is None
    assert source.closest(Time(sec=2), .02) is None
    frame = source.closest(Time(sec=1), .02)
    source.check_camera(camera, frame)
    frame.camera = CameraGeometry([[101, 0, 10], [0, 100, 10], [0, 0, 1]], 20, 20, 'color')
    with pytest.raises(ValueError, match='intrinsics differ'):
        source.check_camera(camera, frame)


def test_detector_rejects_mismatched_grid_without_publishing(scenario):
    from collections import deque
    data, _, params, camera, _ = scenario
    node = BlockDetectorNode.__new__(BlockDetectorNode)
    stamp = Time(sec=100)
    node._p, node._camera, node._processing = params, camera, False
    node._last_color_stamp = node._last_depth_stamp = -np.inf
    node._colors = deque([(data['color_bgr'][0], stamp, camera.frame_id)])
    node._depth_source = DepthSource(Mock())
    node._depth_source._append(np.ones((120, 160)), stamp, camera.frame_id, camera)
    node._diagnostics = dict(processed_frames=0, rejected_frames=0)
    node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=100_000_000_000))
    node.get_logger = lambda: Mock()
    node._publish_tracks = Mock()
    node._get_camera_to_base_tf = Mock()
    node._process()
    assert node._diagnostics['rejected_frames'] == 1
    node._publish_tracks.assert_not_called()
    node._get_camera_to_base_tf.assert_not_called()


def test_fused_sequence_keeps_id_and_visual_speed(scenario):
    from nero7_perception.block_tracker import BlockTracker
    from nero7_perception.conveyor_model import BeltModel
    data, _, params, camera, plane = scenario
    tracker = BlockTracker(BeltModel([0, -1]), {})
    ids = set()
    for i in range(15, len(data['stamps'])):
        detections, _ = detect_blocks(data['color_bgr'][i], data['depth_m'][i], camera,
                                     data['transforms'][i], plane, params)
        assert len(detections) == 1
        tracks = tracker.update(detections, data['stamps'][i])
        assert len(tracks) == 1
        ids.add(tracks[0].id)
    assert ids == {0}
    assert tracks[0].speed_valid
    assert tracks[0].speed == pytest.approx(.028, abs=.003)


def test_missing_truth_does_not_invent_accuracy_or_id_switch_results(scenario):
    data, _, params, _, _ = scenario
    report, _, _ = evaluate_dataset(data, params)
    result = report['modes']['rgb_depth/fixed']
    assert result['position_rmse_m'] is None
    assert result['false_negatives'] is None
    assert result['id_switches'] is None
    assert 'offline' in report['timing_scope']


def test_height_image_matches_transformed_xyz_for_tilted_plane(scenario):
    data, _, _, camera, _ = scenario
    plane = BaseBeltPlane(np.array([-.03, .01, 1.]), .1, 'ransac3d', {})
    expected = plane.vertical_height(camera.cloud(data['depth_m'][0], data['transforms'][0]))
    actual = plane.height_image(data['depth_m'][0], camera, data['transforms'][0])
    np.testing.assert_allclose(actual, expected, atol=1e-14, equal_nan=True)


def test_fused_depth_constraint_rejects_biased_gray_belt_mark():
    data, _ = synthetic_dataset(frames=24, stress=True)
    camera = dataset_camera(data)
    params = load_detection_params(CONFIG)
    params.update(workspace_x=[-1., 1.], plane_roi_px=[20, 20, 300, 220])
    plane, _ = calibrate_dataset(data, camera, params, 'fixed')
    arguments = (data['color_bgr'][22], data['depth_m'][22], camera, data['transforms'][22], plane)
    rgb, _ = detect_blocks(*arguments, dict(params, detection_mode='rgb'))
    fused, _ = detect_blocks(*arguments, dict(params, detection_mode='rgb_depth'))
    assert len(rgb) == 2
    assert len(fused) == 1


def test_unsynchronized_dataset_is_rejected(scenario):
    data, _, params, _, _ = scenario
    data = dict(data, color_stamps=data['color_stamps']+.08)
    with pytest.raises(ValueError, match='timestamps violate'):
        evaluate_dataset(data, params)


def test_new_detection_survives_ros_message_and_controller_prediction(scenario):
    from rclpy.serialization import serialize_message, deserialize_message
    from nero7_msgs.msg import BlockTrackArray
    from nero7_perception.block_tracker import BlockTracker
    from nero7_perception.conveyor_model import BeltModel
    from nero7_control.grasp_controller import TrackView
    from nero7_control.conveyor_prediction import motion_at_contact
    data, truth, params, camera, plane = scenario
    belt = BeltModel([0, -1])
    tracker = BlockTracker(belt, {})
    for index in range(15, len(data['stamps'])):
        detections, _ = detect_blocks(data['color_bgr'][index], data['depth_m'][index], camera,
                                     data['transforms'][index], plane, dict(params, detection_mode='rgb_depth'))
        tracks = tracker.update(detections, data['stamps'][index])
    node = BlockDetectorNode.__new__(BlockDetectorNode)
    node._p, node._belt, node._pub_tracks = dict(base_frame='base_link'), belt, Mock()
    ns = int(round(data['stamps'][-1]*1e9)); seconds, nanoseconds = divmod(ns, 1_000_000_000)
    node._publish_tracks(tracks, Time(sec=seconds, nanosec=nanoseconds))
    message = node._pub_tracks.publish.call_args.args[0]
    received = deserialize_message(serialize_message(message), BlockTrackArray)
    track = TrackView(received.tracks[0])
    assert track.speed_valid
    assert track.belt_z+track.height == pytest.approx(track.centroid_z)
    np.testing.assert_allclose(track.motion_covariance, tracks[0].motion_covariance)
    result = motion_at_contact(track, data['stamps'][-1]+.01, .2, np.zeros(2), np.zeros((2, 2)))
    expected = np.asarray(truth[-1][0]['centroid_base'])[:2]+[0., -.028*.21]
    assert np.linalg.norm(result['position']-expected) < .005


def test_complete_depth_failure_cannot_look_like_fresh_empty_belt(scenario):
    data, _, params, camera, plane = scenario
    with pytest.raises(ValueError, match='do not publish'):
        detect_blocks(data['color_bgr'][0], np.full(data['depth_m'][0].shape, np.nan),
                      camera, data['transforms'][0], plane, params)


@pytest.mark.parametrize('calibration_only', [False, True])
def test_detector_processes_depth_once_and_calibration_never_publishes_tracks(scenario, calibration_only):
    from collections import deque
    data, _, params, camera, plane = scenario
    node = BlockDetectorNode.__new__(BlockDetectorNode)
    stamp = Time(sec=100)
    node._p = dict(params, plane_calibration_enabled=calibration_only)
    node._camera, node._processing = camera, False
    node._last_color_stamp = node._last_depth_stamp = -np.inf
    node._colors = deque([(data['color_bgr'][0], stamp, camera.frame_id)])
    node._depth_source = DepthSource(Mock())
    node._depth_source._append(data['depth_m'][0], stamp, camera.frame_id, camera)
    node._diagnostics = dict(processed_frames=0, rejected_frames=0)
    node._d2c, node._frame_count = (0, 0), 0
    node._plane_context = plane_context(camera, data['transforms'][0], 'base_link')
    node._base_plane, node._pub_debug = plane, None
    node._tracker = Mock(); node._tracker.update.return_value = []
    node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=100_000_000_000))
    node.get_logger = lambda: Mock()
    node._publish_tracks, node._calibrate = Mock(), Mock()
    node._get_camera_to_base_tf = Mock(return_value=(data['transforms'][0, :3, :3], data['transforms'][0, :3, 3]))
    node._process()
    node._process()
    node._colors.append((data['color_bgr'][0], Time(sec=100, nanosec=1_000_000), camera.frame_id))
    node._process()
    if calibration_only:
        node._calibrate.assert_called_once()
        node._publish_tracks.assert_not_called()
        node._tracker.update.assert_not_called()
    else:
        node._publish_tracks.assert_called_once()
        node._tracker.update.assert_called_once()


def test_previous_plane_is_preserved_without_leaving_a_stale_current_result(tmp_path):
    from nero7_perception.base_plane import archive_plane
    target = tmp_path/'belt_plane.yaml'
    target.write_text('previous calibration')
    archived = archive_plane(target)
    assert not target.exists()
    assert Path(archived).read_text() == 'previous calibration'
    assert archive_plane(target) is None
