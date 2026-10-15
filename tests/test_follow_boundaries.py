"""Execute production follow state transitions with explicit driver/plant doubles."""
import time
import json
from collections import deque
from types import SimpleNamespace as NS
from unittest.mock import Mock
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from tests.test_controller_boundaries import production_class
from nero7_control import follow_math as fm
from nero7_control.conveyor_prediction import forecast_motion, motion_at_contact
from nero7_control.belt import BeltModel


STATIC = production_class('src/nero7_control/nero7_control/static_once.py', 'StaticOnce',
                          dict(np=np, time=time))
FOLLOW = production_class('src/nero7_control/nero7_control/conveyor_follow.py', 'ConveyorFollow',
    dict(StaticOnce=STATIC, np=np, time=time, json=json, deque=deque, Rotation=Rotation,
         forecast_motion=forecast_motion, motion_at_contact=motion_at_contact,
         **{name: getattr(fm, name) for name in (
             'vector', 'in_region', 'interception_window', 'smooth_segment',
             'brake_displacement', 'matched', 'resolved_rate', 'TipVelocity', 'flange_in_workspace', 'arm_fault')}))


class CartesianPlant:
    lower, upper = np.full(7, -2.), np.full(7, 2.)
    def fk(self, q):
        T = np.eye(4); T[:3, 3] = q[:3]
        T[:3, :3] = Rotation.from_rotvec(q[3:6]).as_matrix()
        return T
    def jacobian(self, q, tcp):
        return np.eye(6, 7)


def actor_base():
    actor = FOLLOW.__new__(FOLLOW)
    actor.arm, actor.node = Mock(), Mock()
    actor.node._p = dict(dry_run=False)
    actor.clock = 0.
    actor._motion_time = lambda: actor.clock
    actor.node.get_clock.return_value.now.side_effect = lambda: NS(nanoseconds=int(actor.clock*1e9))
    actor.state, actor.started = 'FOLLOW_SYNC', 0.
    actor.motion_issued, actor.failure, actor.track = True, None, NS(id=7)
    actor.report = dict(events=[], diagnostics={})
    actor._preview = Mock()
    actor._execution_enabled = lambda: True
    actor._fresh = lambda: True
    actor.status = NS(err_status=0, arm_status=0)
    actor._observe = Mock()
    actor._request = Mock()
    actor.arm.can_stream_joints.return_value = True
    actor.arm.estop.return_value = None
    return actor


@pytest.mark.parametrize('reason', ['disabled', 'feedback', 'driver', 'endpoint'])
def test_active_follow_failure_stops_without_opening_or_recovery_motion(reason):
    actor = actor_base()
    if reason == 'disabled': actor._execution_enabled = lambda: False
    if reason == 'feedback': actor._fresh = lambda: False
    if reason == 'driver': actor.status.err_status = 1
    if reason == 'endpoint': actor.arm.can_stream_joints.return_value = False
    actor.tick()
    assert actor.state == 'FOLLOW_ABORT'
    actor.arm.estop.assert_called_once()
    actor.arm.stream_joints.assert_not_called()
    actor.arm.move_gripper.assert_not_called()
    actor.arm.move_l.assert_not_called()


def test_first_unconfirmed_image_starts_acquisition_and_inference_upstream():
    actor = actor_base()
    actor.state, actor.motion_issued, actor.track = 'FOLLOW_WAIT_TARGET', False, None
    actor.belt, actor.rejected_until, actor.retry_at = BeltModel(), {}, 0.
    actor.acquisition_at = 0.
    track = NS(id=7, status=0, stamp=0., speed_valid=False, centroid_x=-.8, centroid_y=.2)
    actor.node._tracks = [track]
    actor._acquire(0.)
    assert actor.track.id == 7
    actor._request.assert_called_once_with(0.)
    assert actor.report['diagnostics']['planning_from_first_detection']
    actor.arm.stream_joints.assert_not_called()
    actor.arm.move_j.assert_not_called()


def test_unqualified_visible_target_cannot_be_hidden_by_closure_freeze():
    actor = actor_base()
    actor.state, actor.latency, actor.max_occlusion = 'FOLLOW_CLOSE', .03, .35
    actor.node._tracks = [NS(id=7, status=1, stamp=0., speed_valid=False)]
    actor.frozen = dict(observed_at=0.)
    actor._target = Mock(side_effect=ValueError('Kalman velocity has not converged'))
    with pytest.raises(ValueError, match='converged'):
        actor._live_or_frozen(.1)


def test_closure_prediction_keeps_original_observation_time_and_grows_uncertainty():
    actor = actor_base()
    actor.state, actor.latency, actor.max_occlusion = 'FOLLOW_CLOSE', .03, .35
    actor.node._tracks = []
    actor.frozen = dict(at=.03, observed_at=0., position=np.array([0., 0.]),
        velocity=np.array([.028, 0.]), covariance=np.eye(4)*1e-6,
        process_noise=np.eye(2)*4e-6)
    prediction = actor._live_or_frozen(.25)
    assert prediction['position'] == pytest.approx([.007, 0.])
    assert prediction['covariance'][0, 0] > 1e-6
    assert actor.frozen['observed_at'] == 0.
    with pytest.raises(ValueError, match='occlusion'):
        actor._live_or_frozen(.36)


def test_grip_confirmation_requires_multiple_new_feedback_samples():
    actor = actor_base()
    actor.plan, actor.open_width = dict(close_width=.0205, raw_width=.024), .07
    actor.grip_width = .024
    actor.grip_verified_since, actor.grip_verified_count = None, 0
    actor.grip_counter, actor.grip_blocked_frames = 1, 0
    actor._gripper_fresh = lambda: True
    assert not actor._grasp_verified(.2)
    actor.clock = .2
    assert not actor._grasp_verified(.4)  # Repeated reads of one sample are insufficient.
    actor.grip_counter += 1
    assert not actor._grasp_verified(.4)
    actor.grip_counter += 1
    assert actor._grasp_verified(.4)


def initialize_servo(actor):
    actor.kin, actor.tcp, actor.axis = CartesianPlant(), np.zeros(3), [1., 0., 0.]
    actor.region, actor.margin, actor.belt = [-.4, .4, -.3, .3], .01, BeltModel()
    actor.plan = dict(high_z=.12, low_z=.022, belt_z=-.1, rotation=Rotation.identity(),
                      raw_width=.024, close_width=.0205)
    actor.sync_timeout, actor.descend, actor.close_timeout = 1.5, 2.5, 1.2
    actor.lift_seconds, actor.brake_seconds, actor.lift_offset = 2.5, .6, .1
    actor.latency, actor.open_width = .03, .07
    actor.sigma, actor.position_tol, actor.cross_tol, actor.velocity_tol = 3., .016, .012, .008
    actor.gain, actor.joint_speed, actor.joint_accel = 4., .6, 2.
    actor.linear_speed, actor.angular_speed = .1, .3
    actor.q = np.r_[0., 0., .12, np.zeros(4)]
    actor.q_command, actor.q_velocity = actor.q.copy(), np.zeros(7)
    actor.last_tick = 0.
    actor.velocity_feedback = fm.TipVelocity()
    for stamp in np.arange(-.1, .001, .02):
        actor.velocity_feedback.add(stamp, actor.q[:3])
    actor.sync_since, actor.sync_pose_stamp, actor.sync_frames = None, None, 0
    actor.last_pose_stamp = 0.
    actor.grip_goal, actor.grip_width, actor.grip_counter = .07, .07, 0
    actor.grip_confirmed = False
    actor.grip_verified_count, actor.grip_verified_since, actor.grip_blocked_frames = 0, None, 0
    actor._gripper_fresh = lambda: True
    actor._actual = lambda: (actor.q[:3].copy(), Rotation.identity())
    actor.arm.get_joint_state.side_effect = lambda: actor.q.copy()
    actor._joints = lambda: actor.q.copy()
    actor.node._p['workspace'] = dict(x_min=-.4, x_max=.4, y_min=-.3, y_max=.3, z_min=-.1)
    actor.arm.move_gripper.side_effect = lambda width, **kwargs: setattr(actor, 'grip_goal', width)
    actor._live_or_frozen = lambda now: dict(position=np.array([.028*(now+actor.latency), 0.]),
        velocity=np.array([.028, 0.]), covariance=np.eye(2)*1e-6)


def test_real_state_machine_follows_descends_closes_lifts_brakes_and_holds():
    actor = actor_base()
    initialize_servo(actor)
    seen, closure_commands = set(), []
    for i in range(1, 550):
        actor.clock = i*.02
        # The plant responds with lag, and pose/width feedback are separate samples.
        actor.velocity_feedback.add(actor.clock, actor.q[:3])
        actor.last_pose_stamp = actor.clock
        actor.grip_width += float(np.clip(actor.grip_goal-actor.grip_width, -.004, .004))
        if actor.grip_goal < .024:
            actor.grip_width = max(.024, actor.grip_width)  # Blocked by the test object.
        actor.grip_counter += 1
        previous_state = actor.state
        actor._servo(actor.clock, actor.clock-actor.started)
        seen.add(actor.state)
        if previous_state == 'FOLLOW_CLOSE':
            closure_commands.append(actor.arm.stream_joints.call_args.args[0])
        actor.q += (actor.q_command-actor.q)*.02/(.035+.02)
        if actor.state == 'FOLLOW_DONE':
            break
    assert actor.state == 'FOLLOW_DONE'
    assert {'FOLLOW_DESCEND', 'FOLLOW_CLOSE', 'FOLLOW_LIFT', 'FOLLOW_BRAKE', 'FOLLOW_DONE'} <= seen
    assert len(closure_commands) >= 5
    assert closure_commands[-1][0] > closure_commands[0][0]+.003
    actor.arm.move_gripper.assert_called_once_with(.0205, force=1.5)
    actor.arm.move_j.assert_called_once()
    actor.arm.estop.assert_not_called()
    assert actor.q[2] > .115  # Object is lifted, and never automatically returned/released.


def test_stationary_end_effector_cannot_start_descending():
    actor = actor_base()
    initialize_servo(actor)
    actor.clock, actor.last_tick = .2, .18
    actor._measured = lambda now: (actor.q[:3], Rotation.identity(), np.zeros(3))
    actor._servo(.2, .2)
    assert actor.state == 'FOLLOW_SYNC'
    actor.arm.move_gripper.assert_not_called()
    actor.clock, actor.last_tick = 1.51, 1.49
    with pytest.raises(ValueError, match='synchronize'):
        actor._servo(1.51, 1.51)


def test_speed_mismatch_during_descent_blocks_closure_and_stream():
    actor = actor_base()
    initialize_servo(actor)
    actor.state, actor.clock, actor.last_tick = 'FOLLOW_DESCEND', .5, .48
    actor._measured = lambda now: (actor.q[:3], Rotation.identity(), np.zeros(3))
    with pytest.raises(ValueError, match='Relative'):
        actor._servo(.5, .5)
    actor.arm.move_gripper.assert_not_called()
    actor.arm.stream_joints.assert_not_called()


def test_outside_region_never_reaches_the_close_command():
    actor = actor_base()
    initialize_servo(actor)
    actor.clock, actor.last_tick = .2, .18
    actor.q[0] = .5
    actor._measured = lambda now: (actor.q[:3], Rotation.identity(), np.zeros(3))
    with pytest.raises(ValueError, match='region'):
        actor._servo(.2, .2)
    actor.arm.move_gripper.assert_not_called()
    actor.arm.stream_joints.assert_not_called()

SIM = production_class('src/nero7_sim/nero7_sim/driver.py', 'SimDriver',
                       dict(Node=object, np=np))


def test_sim_driver_stream_is_latest_only_and_never_pretends_to_be_feedback():
    driver = SIM.__new__(SIM)
    driver.enabled, driver.fault = True, 0
    driver.kin = CartesianPlant()
    driver.seconds = lambda: 1.
    driver.planning, driver.trajectory = None, None
    driver.actual_q = np.full(7, .003)
    driver.move_js(NS(name=['joint%d' % i for i in range(1, 8)], position=[.01]*7))
    driver.move_js(NS(name=['joint%d' % i for i in range(1, 8)], position=[.02]*7))
    assert driver.stream_target == pytest.approx([.02]*7)
    assert driver.actual_q == pytest.approx([.003]*7)
    assert driver.trajectory is None
    driver.move_js(NS(name=['joint1'], position=[.1]))
    assert driver.fault == 1

@pytest.mark.parametrize('field', ['joint_angle_limit', 'communication_status_joint'])
def test_joint_fault_flags_stop_even_when_numeric_err_status_is_zero(field):
    actor = actor_base()
    setattr(actor.status, field, [False, True]+[False]*5)
    actor.tick()
    assert actor.state == 'FOLLOW_ABORT'
    actor.arm.estop.assert_called_once()
    actor.arm.stream_joints.assert_not_called()


def test_live_flange_guard_checks_tcp_offset_in_addition_to_tip_region():
    actor = actor_base()
    initialize_servo(actor)
    actor.clock, actor.last_tick = .2, .18
    actor.tcp = np.array([.5, 0., 0.])
    actor._measured = lambda now: (np.array([0., 0., .12]), Rotation.identity(), np.array([.028, 0, 0]))
    with pytest.raises(ValueError, match='flange'):
        actor._servo(.2, .2)
    actor.arm.stream_joints.assert_not_called()


def test_early_speed_match_waits_for_the_preflight_descent_time():
    actor = actor_base()
    initialize_servo(actor)
    actor.clock, actor.last_tick = .4, .38
    actor.q[0] = .028*.4
    actor.q_command = actor.q.copy()
    actor.sync_since, actor.sync_frames, actor.sync_pose_stamp = .1, 12, .38
    actor.last_pose_stamp = .4
    actor._measured = lambda now: (actor.q[:3].copy(), Rotation.identity(), np.array([.028, 0., 0.]))
    actor._servo(.4, .4)
    assert actor.state == 'FOLLOW_SYNC'
    actor.arm.move_gripper.assert_not_called()


def test_early_grip_confirmation_keeps_following_until_preflight_lift_time():
    actor = actor_base()
    initialize_servo(actor)
    actor.state, actor.clock, actor.last_tick = 'FOLLOW_CLOSE', .5, .48
    actor.q[:3] = [.028*.5, 0., actor.plan['low_z']]
    actor.q_command = actor.q.copy()
    actor.grip_width = .024
    actor._grasp_verified = Mock(return_value=True)
    actor._measured = lambda now: (actor.q[:3].copy(), Rotation.identity(), np.array([.028, 0., 0.]))
    actor._servo(.5, .5)
    assert actor.grip_confirmed and actor.state == 'FOLLOW_CLOSE'
    actor.clock, actor.last_tick = 1.21, 1.19
    actor.q[0] = .028*1.21
    actor.q_command = actor.q.copy()
    actor._live_or_frozen = Mock(side_effect=AssertionError('Captured object must not need visibility'))
    actor._servo(1.21, 1.21)
    assert actor.state == 'FOLLOW_LIFT'
    actor._live_or_frozen.assert_not_called()
