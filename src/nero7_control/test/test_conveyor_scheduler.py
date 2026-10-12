"""Interception state transitions with real predictions and no ROS executor/robot."""
import time
from concurrent.futures import Future
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from nero7_control.belt import BeltModel
from nero7_control.conveyor_once import ConveyorOnce


def make_track(remaining=6.2, now=100., **changes):
    values = dict(id=7, status=1, speed_valid=True, stamp=now, motion_stamp=now,
                  centroid_x=-.4, centroid_y=.2+.028*remaining, centroid_z=-.065,
                  observed_position=np.array([-.4, .2+.028*remaining, -.065]),
                  velocity=np.array([0., -.028, 0.]),
                  motion_covariance=np.diag([1e-7, 1e-7, 1e-8, 1e-8]),
                  motion_process_noise=np.diag([1e-9, 1e-9]),
                  belt_z=-.1, u_px=500., v_px=400.)
    values.update(changes)
    return NS(**values)


@pytest.fixture
def actor(tmp_path):
    a = ConveyorOnce.__new__(ConveyorOnce)
    a.node = Mock()
    a.node._p = dict(dry_run=False)
    a.node.get_parameter.return_value = NS(value=False)
    a.node.get_clock.return_value.now.return_value = NS(nanoseconds=100_000_000_000)
    a.arm = Mock()
    a.arm.get_joint_state.return_value = np.zeros(7)
    a.arm.get_gripper_width.return_value = .07
    a.enable = Mock()
    a.state = 'CONVEYOR_WAIT_START'
    a.started = time.monotonic()
    a.belt = BeltModel([0., -1.], [-.4, 0.], -.2)
    a.belt_z = -.1
    a.approach_seconds, a.prepare_seconds = 2., 1.5
    a.descend_seconds, a.close_seconds, a.contact_seconds = 2.9, .26, .24
    a.horizon = 3.14
    a.start_margin, a.trigger_margin = .8, .12
    a.timing_std, a.sigma = .05, 3.
    a.along_tolerance, a.cross_tolerance = .025, .016
    a.timing_max_age = 3600.
    a.stable_count, a.stable_stamp, a.stable_frames = 0, None, 3
    a.offset_xy, a.offset_covariance = np.zeros(2), np.eye(2)*1e-7
    a.axis = [1., 0., 0.]
    a.open_width = .07
    a.calibration_goal, a.stroke = .0205, .078
    a.calibration_yaw = 1.444185508449333
    a.calibration_rotation = Rotation.identity()
    a.timing_verified, a.driver_speed = True, 20
    a.report = dict(events=[], predictions=[], timing_verified=True,
                    driver_speed_percent=20, calibrated_at_unix=time.time())
    a.timing_file = str(tmp_path/'timing.json')
    a.motion_issued, a.failure, a.calibration_only = False, None, False
    a._fresh = lambda: True
    a._driver_profile = lambda: True
    a._actual = lambda: (np.zeros(3), Rotation.identity())
    a.status = NS(arm_status=0, motion_status=0, err_status=0)
    a.plan = dict(grasp=np.array([-.4, .2, -.08]), above=np.array([-.4, .2, -.002]),
                  rotation=Rotation.identity(), raw_width=.024, close_width=.0204,
                  joints=[np.zeros(7)]*3, seed_joints=np.zeros(7),
                  seed_flange=np.zeros(3), seed_rotation=Rotation.identity())
    a.track = make_track()
    a.node._tracks = [a.track]
    a.roi, a.offset, a.clearance = [180, 355, 1060, 475], .018, .012
    a.history, a.candidates, a.rejected_until = {}, {}, {}
    a.acquisition_id, a.acquisition_started, a.planning_reserve = None, 0., 2.5
    a.calibration_empty_since = None
    a._reuse_verified_timing = lambda: False
    a._make_plan = Mock(return_value=a.plan)
    a.planner = Mock()
    a.calibration_future = None
    return a


def update(a, remaining=7., now=100., **changes):
    a.track = make_track(remaining, now, **changes)
    a.node._tracks = [a.track]
    a.node.get_clock.return_value.now.return_value = NS(nanoseconds=int(now*1e9))


def test_ik_completion_waits_without_moving(actor):
    update(actor, 10.)
    future = Future(); future.set_result(actor.plan)
    actor.planning_future = future
    actor._planning_complete(100.)
    assert actor.state == 'CONVEYOR_WAIT_START'
    actor.arm.move_j.assert_not_called()
    actor.enable.call_async.assert_not_called()


@pytest.mark.parametrize('remaining', [12., 6.5, -1.])
def test_outside_start_window_does_not_enable_or_move(actor, remaining):
    actor.node.get_parameter.return_value.value = True
    update(actor, remaining)
    for _ in range(5):
        actor._wait_start(100.)
    actor.enable.call_async.assert_not_called()
    actor.arm.move_j.assert_not_called()


def test_start_requires_distinct_frames(actor):
    actor.node.get_parameter.return_value.value = True
    update(actor)
    for _ in range(20):
        actor._wait_start(100.)
    assert actor.stable_count == 1
    actor.enable.call_async.assert_not_called()
    for i in [1, 2]:
        update(actor, now=100.+i/30)
        actor._wait_start(100.+i/30)
    assert actor.state == 'CONVEYOR_START_ENABLE'
    actor.enable.call_async.assert_called_once()
    actor.arm.move_j.assert_not_called()


@pytest.mark.parametrize('dry_run', [False, True])
def test_preview_never_sends_arm_or_gripper_commands(actor, dry_run):
    actor.node._p['dry_run'] = dry_run
    actor.node.get_parameter.return_value.value = dry_run
    for i in range(3):
        update(actor, now=100.+i/30)
        actor._wait_start(100.+i/30)
    assert actor.report['preview_would_start']
    actor.enable.call_async.assert_not_called()
    actor.arm.move_j.assert_not_called()
    actor.arm.move_gripper.assert_not_called()


@pytest.mark.parametrize('change', [dict(speed_valid=False), dict(status=2), dict(stamp=99.),
                                    dict(motion_covariance=np.eye(4)*.01),
                                    dict(centroid_x=-.36)])
def test_invalid_motion_or_uncertainty_prevents_start(actor, change):
    actor.node.get_parameter.return_value.value = True
    for i in range(3):
        update(actor, now=100.+i/30, **change)
        actor._wait_start(100.+i/30)
    actor.enable.call_async.assert_not_called()
    actor.arm.move_j.assert_not_called()


def test_expired_timing_cannot_start_or_calibrate_automatically(actor):
    actor.node.get_parameter.return_value.value = True
    actor.report['calibrated_at_unix'] = time.time()-4000.
    for i in range(3):
        update(actor, now=100.+i/30)
        actor._wait_start(100.+i/30)
    assert actor.state == 'CONVEYOR_NEED_CALIBRATION'
    actor.enable.call_async.assert_not_called()


def test_pose_change_invalidates_preflight(actor):
    actor.node.get_parameter.return_value.value = True
    actor.arm.get_joint_state.return_value = np.full(7, .2)
    for i in range(3):
        update(actor, now=100.+i/30)
        actor._wait_start(100.+i/30)
    assert actor.state == 'CONVEYOR_WAIT_TARGET'
    assert 'preflight' in actor.report['last_skip']
    actor.enable.call_async.assert_not_called()


def test_missing_timings_waits_without_empty_calibration(actor):
    actor.state, actor.timing_verified = 'CONVEYOR_WAIT_TARGET', False
    actor.node.get_parameter.return_value.value = True
    actor.tick()
    assert actor.state == 'CONVEYOR_NEED_CALIBRATION'
    actor._make_plan.assert_not_called()
    actor.enable.call_async.assert_not_called()


def test_calibration_yields_after_slow_preflight_and_rechecks_feedback(actor):
    actor.state, actor.calibration_only, actor.plan = 'CONVEYOR_CALIB_READY', True, None
    actor.node.get_parameter.return_value.value = True
    actor._calibration_clear = lambda now: True
    actor._calibration_plan = Mock(side_effect=lambda: setattr(actor, 'plan', {}))
    actor._seed_matches = lambda: True
    future = Future()
    actor.planner.submit.return_value = future
    actor.tick()
    actor.planner.submit.assert_called_once_with(actor._calibration_plan)
    actor.tick()
    actor.enable.call_async.assert_not_called()
    actor._calibration_plan()
    future.set_result(None)
    actor.tick()
    actor.enable.call_async.assert_not_called()
    actor._fresh = lambda: False
    actor.tick()
    assert actor.state == 'CONVEYOR_CALIB_READY'
    actor.enable.call_async.assert_not_called()
    actor._fresh = lambda: True
    actor.tick()
    actor.enable.call_async.assert_called_once()
    assert actor.state == 'CONVEYOR_CALIB_ENABLE'


def test_arm_change_after_calibration_preflight_requires_new_plan(actor):
    actor.state, actor.calibration_only = 'CONVEYOR_CALIB_READY', True
    actor.node.get_parameter.return_value.value = True
    actor._calibration_clear = lambda now: True
    actor.arm.get_joint_state.return_value = np.full(7, .2)
    actor.tick()
    assert actor.plan is None
    assert not actor.motion_issued
    actor.enable.call_async.assert_not_called()


def test_gate_requires_distinct_fresh_capture_predictions(actor):
    actor.state = 'CONVEYOR_WAIT_GATE'
    actor.node.get_parameter.return_value.value = True
    actor._motion = Mock()
    update(actor, actor.horizon+.04)
    for _ in range(10):
        actor._wait_gate(100.)
    actor._motion.assert_not_called()
    for i in [1, 2]:
        update(actor, actor.horizon+.04, 100.+i/30)
        actor._wait_gate(100.+i/30)
    actor._motion.assert_called_once()
    assert actor._motion.call_args.args[1] == 'CONVEYOR_DESCEND'


def test_high_capture_uncertainty_blocks_descent(actor):
    actor.state = 'CONVEYOR_WAIT_GATE'
    actor.node.get_parameter.return_value.value = True
    actor._motion = Mock()
    for i in range(3):
        update(actor, actor.horizon+.04, 100.+i/30, motion_covariance=np.eye(4)*.01)
        actor._wait_gate(100.+i/30)
    actor._motion.assert_not_called()


def test_slow_approach_aborts_before_closure(actor):
    actor.state, actor.motion_issued = 'CONVEYOR_APPROACH', True
    actor.startup_started = time.monotonic()-5.
    actor.node.get_parameter.return_value.value = True
    update(actor, 4.)
    actor.tick()
    assert actor.state == 'CONVEYOR_ABORT'
    actor.arm.estop.assert_called_once()
    actor.arm.move_gripper.assert_not_called()


def test_candidate_offset_is_local_until_target_selected(actor):
    update(actor, 10.)
    for i in range(3):
        stamp = 100.+i/30
        actor.history[7] = [(stamp, np.array([-.4, .48, -.065]))]
        result = dict(stamp=stamp, center=np.array([-.39, .48, -.065]), yaw=.2,
                      raw_width_mm=24., width_mm=20.4)
        candidate = actor._candidate(result, actor.track, stamp)
    assert candidate is not None
    np.testing.assert_allclose(candidate['offset_xy'], [.01, 0.])
    np.testing.assert_allclose(actor.offset_xy, [0., 0.])


def test_acquisition_keeps_one_owner_for_latest_only_inference(actor):
    actor.state = 'CONVEYOR_WAIT_TARGET'
    actor.node._tracks = [make_track(12., id=1), make_track(13., id=2)]
    actor.node._inference.latest.return_value = None
    actor._acquire(100.)
    assert actor.acquisition_id == 1
    assert actor.node._fsm.target_id == 1
    actor.node._tracks.reverse()
    actor._acquire(100.)
    assert actor.node._fsm.target_id == 1
    assert actor.node._maybe_request_inference.call_count == 2


def test_calibration_requires_fresh_empty_frames_for_dwell(actor):
    actor.node._tracks = []
    for now in [100., 100.3, 100.7]:
        stamp = NS(sec=int(now), nanosec=int((now-int(now))*1e9))
        for name in ('_tracks_stamp', '_color_stamp', '_depth_stamp'):
            setattr(actor.node, name, stamp)
        clear = actor._calibration_clear(now)
    assert clear
    actor.node._tracks = [make_track()]
    assert not actor._calibration_clear(100.7)
    assert actor.calibration_empty_since is None


def test_stale_empty_detector_does_not_allow_calibration(actor):
    actor.node._tracks = []
    for name in ('_tracks_stamp', '_color_stamp', '_depth_stamp'):
        setattr(actor.node, name, NS(sec=99, nanosec=0))
    assert not actor._calibration_clear(100.)


def test_verified_calibration_only_finishes_without_starting_pick(actor):
    actor.state, actor.calibration_only = 'CONVEYOR_CALIB_READY', True
    actor.node.get_parameter.return_value.value = True
    actor._reuse_verified_timing = lambda: True
    actor._calibration_clear = lambda now: True
    actor.tick()
    assert actor.state == 'CONVEYOR_CALIB_DONE'
    actor.enable.call_async.assert_not_called()


def timing_record(a):
    a.tcp = np.array([.183, .0076, -.0256])
    a.calibration = dict(grasp_rpy=[0., 0., 0.])
    xy = a.belt.point_of(a.belt.gate_s)
    a.previous_timing = dict(timing_verified=True, driver_speed_percent=20,
                             calibrated_at_unix=time.time(), tcp_flange_m=a.tcp.tolist(),
                             calibration_tip_high_m=[*xy, a.belt_z+.122],
                             calibration_tip_low_m=[*xy, a.belt_z+.044],
                             calibration_rpy=[0., 0., 1.444185508449333])


def test_timing_reuse_does_not_require_an_object_plan(actor):
    timing_record(actor)
    actor.plan = None
    assert ConveyorOnce._reuse_verified_timing(actor)
    assert actor.report['descend_seconds'] == 2.9
    assert actor.report['empty_close_seconds'] == .26
    assert actor.calibration_rotation is not None


@pytest.mark.parametrize('field,value', [('timing_verified', False),
                                        ('driver_speed_percent', 50),
                                        ('calibrated_at_unix', 0),
                                        ('tcp_flange_m', []),
                                        ('calibration_tip_low_m', [-.4, .2, 0.]),
                                        ('calibration_rpy', [0., 0., 0.])])
def test_incompatible_timing_record_cannot_enable_execution(actor, field, value):
    timing_record(actor)
    actor.previous_timing[field] = value
    assert not ConveyorOnce._reuse_verified_timing(actor)


@pytest.mark.parametrize('field', ['stamp', 'yaw', 'raw_width_mm', 'width_mm'])
def test_nonfinite_network_result_cannot_be_selected(actor, field):
    result = dict(stamp=100., yaw=0., raw_width_mm=24., width_mm=20.)
    result[field] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite'):
        actor._candidate(result, actor.track, 100.)


@pytest.mark.parametrize('state', ['CONVEYOR_WAIT_GATE', 'CONVEYOR_WAIT_TARGET',
                                  'CONVEYOR_START_ENABLE', 'CONVEYOR_DESCEND'])
def test_disabling_after_any_accepted_motion_requests_stop(actor, state):
    actor.state, actor.motion_issued = state, True
    actor.node.get_parameter.return_value.value = False
    actor.tick()
    assert actor.state == 'CONVEYOR_ABORT'
    actor.arm.estop.assert_called_once()
    actor.arm.move_l.assert_not_called()
    actor.arm.move_gripper.assert_not_called()
