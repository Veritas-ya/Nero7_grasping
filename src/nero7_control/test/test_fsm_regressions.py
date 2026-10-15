"""Regression scenarios for the actual continuous FSM, without ROS or hardware."""
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from nero7_control.belt import BeltModel
from nero7_control.fsm import GraspFsm, IDLE, TRACK, WAIT, PLACE, HALT


def machine(**changes):
    config = dict(bin_pre_joints=[0.]*7, bin_drop_pose=[.1, .2, .3, 0., 0., 0.])
    config.update(changes)
    return GraspFsm(BeltModel([1., 0.], [0., 0.], .35), config)


def track(**changes):
    data = dict(id=4, status=1, speed_ok=True, speed=.028, stamp=10.,
                motion_stamp=10., s=.2, centroid_x=.2, centroid_y=.02,
                centroid_z=.04, belt_z=0., height=.04, u_px=200., v_px=100.)
    data.update(changes)
    return NS(**data)


def inputs(**changes):
    data = dict(now=10., tracks=[track()], grasp=None, arm_done=False,
                gripper_width=.03, joint_state=[0.]*7)
    data.update(changes)
    return data


def test_placement_never_advances_while_motion_is_busy():
    with patch('nero7_control.fsm.time.monotonic', return_value=0.) as clock:
        actor = machine()
        actor._enter(PLACE)
        commands = actor.tick(inputs())
        assert [cmd['type'] for cmd in commands] == ['move_j']
        for i in range(1, 100):
            clock.return_value = i*.02
            assert actor.tick(inputs()) == []
        assert actor.state == PLACE and actor.successes == 0 and actor._step == 1
        clock.return_value = 2.1
        assert actor.tick(inputs(arm_done=True))[0]['type'] == 'move_l'
        for i in range(110, 160):
            clock.return_value = i*.02
            assert actor.tick(inputs()) == []
        clock.return_value = 3.3
        assert actor.tick(inputs(arm_done=True))[0]['type'] == 'gripper'
        clock.return_value = 3.7
        actor.tick(inputs(gripper_width=.03))
        assert actor.successes == 0
        actor.tick(inputs(gripper_width=.07))
        assert actor.successes == 1 and actor.state == IDLE


@pytest.mark.parametrize('bad', [[], None, [0.]*5, [np.nan]*6])
def test_invalid_placement_stops_without_releasing(bad):
    actor = machine(bin_drop_pose=bad)
    actor._enter(PLACE)
    commands = actor.tick(inputs())
    assert actor.state == HALT
    assert not any(cmd['type'] in ('move_j', 'move_l', 'gripper') for cmd in commands)


@pytest.mark.parametrize('change', [dict(speed=0., speed_ok=False), dict(speed=np.nan),
                                   dict(stamp=9.), dict(stamp=11.), dict(status=2)])
def test_invalid_selected_velocity_cannot_divide_or_trigger(change):
    actor = machine()
    actor.target_id = 4
    actor._enter(TRACK)
    assert actor.tick(inputs(tracks=[track(**change)])) == []
    assert actor.state == IDLE


def test_loss_after_approach_requests_stop():
    actor = machine()
    actor.target_id = 4
    actor._enter(WAIT)
    commands = actor.tick(inputs(tracks=[]))
    assert actor.state == HALT
    assert any(cmd['type'] == 'estop' for cmd in commands)


def test_grasp_is_transported_to_gate_and_tcp_is_converted_to_flange():
    tcp = [.183, .008, -.026]
    actor = machine(gripper_tcp=tcp, grasp_rpy=[1.179, 1.235, 2.725])
    actor.grasp = dict(status=0, center=np.array([.2, .02, .04]), yaw=.7)
    pose = actor._grasp_pose(track())
    rotation = Rotation.from_euler('xyz', pose[3:])
    tip = np.asarray(pose[:3])+rotation.apply(tcp)
    np.testing.assert_allclose(tip[:2], [.35, .02], atol=1e-10)
    above = actor._gate_above_pose(track())
    np.testing.assert_allclose(above[:2], pose[:2])
    assert tip[2] >= actor.cfg['grasp_belt_clearance']


def test_latest_missing_expired_or_wrong_owner_candidate_is_cleared():
    actor = machine()
    actor.target_id = 4
    actor._enter(WAIT)
    base = dict(status=0, stamp=10., target_id=4, center=np.array([.2, .02, .04]), yaw=0.)
    actor.tick(inputs(grasp=base))
    assert actor.grasp is base
    for result in [None, dict(base, stamp=9.), dict(base, target_id=8)]:
        actor.tick(inputs(grasp=result))
        assert actor.grasp is None


def test_predicted_track_is_not_extrapolated_twice():
    actor = machine()
    predicted = track(stamp=9.9, motion_stamp=10., s=.2028)
    assert actor._pred_s(predicted, inputs(now=10.05)) == pytest.approx(.2042)
