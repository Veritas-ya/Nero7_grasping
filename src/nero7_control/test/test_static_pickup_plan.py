"""Pickup acceptance is independent of placement; transfer still needs preflight."""
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from nero7_control.fixed_bin import plan_fixed_bin
from nero7_control.static_once import StaticOnce


class CartesianKinematics:
    """Deterministic pose model for testing planner/state boundaries without ROS."""

    lower = np.full(7, -3., dtype=float)
    upper = np.full(7, 3., dtype=float)

    def fk(self, joints):
        transform = np.eye(4)
        transform[:3, 3] = joints[:3]
        transform[:3, :3] = Rotation.from_rotvec(joints[3:6]).as_matrix()
        return transform

    def solve(self, position, rotation, seed):
        return np.r_[position, rotation.as_rotvec(), seed[6]]


@pytest.fixture
def pickup(tmp_path):
    actor = StaticOnce.__new__(StaticOnce)
    actor.node, actor.arm = Mock(), Mock()
    actor.node._p = dict(workspace=dict(x_min=-.5, x_max=.45,
                                      y_min=-.35, y_max=.35, z_min=-.2))
    actor.kin = CartesianKinematics()
    actor.tcp = np.zeros(3)
    actor.calibration = dict(grasp_rpy=[0., 0., 0.])
    actor.axis = [1., 0., 0.]
    actor.offset, actor.clearance, actor.open_width = .018, .012, .070
    actor.center_source = 'track_centroid'
    actor.max_network_track_xy = .04
    actor.plan_file = str(tmp_path/'pickup.json')
    seed = np.array([-.36, .2, .15, 0., 0., 0., 0.])
    actor.arm.get_joint_state.return_value = seed.tolist()
    actor._actual = lambda: (seed[:3].copy(), Rotation.identity())
    actor.bin_profile = dict(configured=True)  # No placement geometry needed here.
    actor.bin_workspace = dict(actor.node._p['workspace'])
    actor.rrt = Mock()
    actor._motion_time = lambda: 100.
    actor.motion_issued = False
    actor.gripper_feedback = Mock()
    actor.gripper_feedback.reading.return_value = .04
    track = SimpleNamespace(id=7, centroid_x=-.36, centroid_y=.2,
                            centroid_z=-.09, belt_z=-.125)
    result = dict(center=[-.36, .2, -.09], yaw=0., raw_width_mm=30.,
                  width_mm=25., score=.8)
    return actor, result, track


def test_pickup_accepts_without_evaluating_bin_route(pickup):
    actor, result, track = pickup
    with patch('nero7_control.static_once.fixed_bin_targets',
               side_effect=ValueError('Bin route unreachable')) as placement:
        plan = actor._make_plan(result, track)
    placement.assert_not_called()
    assert plan['placement'] is None
    for target, joints in zip((plan['above'], plan['grasp'], plan['lift']), plan['joints']):
        np.testing.assert_allclose(actor.kin.fk(joints)[:3, 3], target)
    assert plan['lift'][2] == pytest.approx(plan['grasp'][2]+.10)
    saved = json.loads(open(actor.plan_file).read())
    assert saved['fixed_bin'] is None
    assert saved['placement_status'] == 'deferred_until_after_lift'
    actor.arm.move_j.assert_not_called()
    actor.arm.move_l.assert_not_called()
    actor.arm.move_gripper.assert_not_called()


def test_network_center_source_uses_complete_grconvnet_pose(pickup):
    actor, result, track = pickup
    actor.center_source = 'network'
    result['center'] = [-.38, .225, -.092]
    plan = actor._make_plan(result, track)
    np.testing.assert_allclose(plan['center'], result['center'])
    saved = json.loads(open(actor.plan_file).read())
    assert saved['center_source'] == 'network'
    assert saved['source'] == 'GR-ConvNet'
    assert saved['network_track_xy_delta_mm'] == pytest.approx(
        np.linalg.norm([-.02, .025])*1000.)


def test_network_center_must_agree_with_tracked_object(pickup):
    actor, result, track = pickup
    actor.center_source = 'network'
    result['center'] = [-.405, .24, -.09]
    with pytest.raises(ValueError, match='outside target neighborhood'):
        actor._make_plan(result, track)


@pytest.mark.parametrize('stage', ['above', 'grasp', 'lift'])
def test_unreachable_pickup_pose_still_rejects(pickup, stage):
    actor, result, track = pickup
    grasp_height = max(track.centroid_z-.018, track.belt_z+actor.clearance)
    height = dict(above=track.centroid_z+.12,
                  grasp=grasp_height, lift=grasp_height+.10)[stage]
    solve = actor.kin.solve

    def limited_solve(position, rotation, seed):
        if np.isclose(position[2], height):
            raise ValueError('Unreachable '+stage)
        return solve(position, rotation, seed)

    actor.kin.solve = limited_solve
    with pytest.raises(ValueError, match='No verified pickup path.*above/grasp/lift'):
        actor._make_plan(result, track)


def test_rrtconnect_uses_measured_joints_after_lift(pickup):
    actor, result, track = pickup
    actor.plan = actor._make_plan(result, track)
    measured = actor.plan['joints'][2].copy()
    measured[6] += .2  # Cartesian execution may select another redundant IK seed.
    actor._require_bin_holding = Mock()
    actor._require_bin_pose = Mock(return_value=measured)
    target = np.array([.1, -.2, .3, .4, -.5, .6, -.7])
    placement = dict(above_joints=target)
    with patch('nero7_control.static_once.fixed_bin_targets', return_value=placement):
        actor._begin_rrtconnect()
    actor._require_bin_pose.assert_called_once_with(
        actor.plan['lift'], actor.plan['rotation'])
    np.testing.assert_allclose(actor.rrt.begin.call_args.args[0], measured)
    np.testing.assert_allclose(actor.rrt.begin.call_args.args[1], target)
    assert actor.rrt.begin.call_args.kwargs['gripper_width'] == actor.gripper_feedback.reading.return_value
    assert actor.plan['placement'] is placement
    assert actor.state == 'STATIC_RRTCONNECT_BIN_PLAN'
    actor.arm.move_j.assert_not_called()
    actor.arm.move_l.assert_not_called()


def test_failed_deferred_placement_stops_before_transfer(pickup):
    actor, result, track = pickup
    actor.plan = actor._make_plan(result, track)
    actor.state, actor.started = 'STATIC_RRTCONNECT_BIN_PLAN', 100.
    actor.motion_issued = True  # Pickup has already executed.
    actor._fresh = lambda: True
    actor.status = SimpleNamespace(err_status=0, arm_status=0)
    actor.node.get_clock().now.return_value = SimpleNamespace(nanoseconds=100_000_000_000)
    actor._require_bin_holding = Mock()
    actor.rrt.busy = True
    actor.rrt.poll.side_effect = ValueError('RRTConnect route unreachable')
    with patch('nero7_control.static_once.fixed_bin_profile_current', return_value=True):
        actor.tick()
    assert actor.state == 'STATIC_ABORT'
    assert actor.failure == 'RRTConnect route unreachable'
    actor.arm.move_j.assert_not_called()
    actor.arm.move_l.assert_not_called()
    actor.arm.move_gripper.assert_not_called()
    actor.arm.estop.assert_called_once()


def test_deferred_bin_plan_adds_carry_raise_after_low_pickup_lift(tmp_path):
    kin = CartesianKinematics()
    profile_path = tmp_path/'fixed-bin.yaml'
    profile = dict(profile_path=str(profile_path), sha256='test', pose_type='joints',
                   above_joints_rad=[0., 0., .10, 0., 0., 0., 0.],
                   release_joints_rad=[0., 0., .02, 0., 0., 0., 0.],
                   via_joints_rad=[], return_path_joints_rad=[])
    lift = np.array([.2, .1, 0., 0., 0., 0., 0.])
    workspace = dict(x_min=-.5, x_max=.5, y_min=-.5, y_max=.5, z_min=-.2)
    placement = plan_fixed_bin(kin, lift, np.zeros(3), profile, -.1, workspace)
    assert placement['pickup_raise_tcp_m'] is not None
    assert placement['pickup_raise_tcp_m'][2] == pytest.approx(.10)
    np.testing.assert_allclose(placement['transfer_joints'][0][:3], [.2, .1, .10])
    np.testing.assert_allclose(placement['transfer_joints'][-1][:3], [0., 0., .10])
    assert placement['minimum_transfer_tip_z_m'] >= -.05


def test_fixed_initial_pose_is_the_final_return_waypoint(tmp_path):
    kin = CartesianKinematics()
    initial = [.2, 0., .10, 0., 0., 0., 0.]
    profile = dict(profile_path=str(tmp_path/'fixed-bin.yaml'), sha256='test',
                   pose_type='joints',
                   above_joints_rad=[0., 0., .10, 0., 0., 0., 0.],
                   release_joints_rad=[0., 0., .02, 0., 0., 0., 0.],
                   via_joints_rad=[],
                   return_path_joints_rad=[[.1, 0., .10, 0., 0., 0., 0.]],
                   initial_joints_rad=initial)
    lift = np.array([.2, .1, .10, 0., 0., 0., 0.])
    workspace = dict(x_min=-.5, x_max=.5, y_min=-.5, y_max=.5, z_min=-.2)
    placement = plan_fixed_bin(kin, lift, np.zeros(3), profile, -.1, workspace)
    assert len(placement['return_joints']) == 2
    np.testing.assert_allclose(placement['return_joints'][-1], initial)
    np.testing.assert_allclose(placement['initial_joints'], initial)


def release_actor(pickup, elapsed=2.):
    actor, _, _ = pickup
    rotation = Rotation.identity()
    actor.plan = dict(placement=dict(release=np.array([0., 0., 0.]),
                                     above=np.array([0., 0., .15]),
                                     rotation=rotation))
    actor.bin_profile = dict(profile_path='unused', sha256='unused')
    actor.state, actor.started = 'STATIC_BIN_RELEASE', 0.
    actor._motion_time = lambda: elapsed
    actor._fresh = lambda: True
    actor.status = SimpleNamespace(err_status=0, arm_status=0, motion_status=1)
    actor.node.get_clock().now.return_value = SimpleNamespace(nanoseconds=100_000_000_000)
    actor.gripper_feedback = Mock()
    actor.gripper_feedback.opening_confirmed.return_value = True
    actor._require_bin_pose = Mock()
    actor._motion = Mock()
    return actor


def test_release_retracts_when_only_gripper_keeps_motion_status_nonzero(pickup):
    actor = release_actor(pickup)
    placement = actor.plan['placement']
    with patch('nero7_control.static_once.fixed_bin_profile_current', return_value=True):
        actor.tick()
    actor._require_bin_pose.assert_called_once_with(
        placement['release'], placement['rotation'], require_motion_status=False)
    actor._motion.assert_called_once_with(
        placement['above'], 'STATIC_BIN_RETRACT', rotation=placement['rotation'])


def test_release_retracts_upward_once_arm_status_settles(pickup):
    actor = release_actor(pickup)
    actor.status.motion_status = 0
    placement = actor.plan['placement']
    with patch('nero7_control.static_once.fixed_bin_profile_current', return_value=True):
        actor.tick()
    actor._require_bin_pose.assert_called_once_with(
        placement['release'], placement['rotation'], require_motion_status=False)
    actor._motion.assert_called_once_with(
        placement['above'], 'STATIC_BIN_RETRACT', rotation=placement['rotation'])


def test_release_pose_still_checks_fresh_joint_fk_when_shared_status_is_one(pickup):
    actor, _, _ = pickup
    joints = np.array([.1, .2, .3, 0., 0., 0., 0.])
    actor.arm.get_joint_state.return_value = joints.tolist()
    actor.status = SimpleNamespace(motion_status=1)
    actor._actual = lambda: (joints[:3].copy(), Rotation.identity())
    with pytest.raises(ValueError, match='Arm must be settled'):
        actor._require_bin_pose(joints[:3], Rotation.identity())
    measured = actor._require_bin_pose(
        joints[:3], Rotation.identity(), require_motion_status=False)
    np.testing.assert_allclose(measured, joints)
