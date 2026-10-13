"""Geometry and closed-loop plant tests; no ROS, CAN or GPU."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from nero7_control.kinematics import Kinematics
from nero7_control.follow_math import (interception_window, in_region, region_box,
    smooth_segment, brake_displacement, TipVelocity, matched, resolved_rate,
    preflight_follow)


class CartesianPlant:
    """Ideal 6-axis Cartesian mechanism + one redundant joint; test double only."""
    lower = np.full(7, -2.)
    upper = np.full(7, 2.)

    def fk(self, q):
        T = np.eye(4)
        T[:3, 3] = q[:3]
        T[:3, :3] = Rotation.from_rotvec(q[3:6]).as_matrix()
        return T

    def jacobian(self, q, tcp=None):
        return np.eye(6, 7)

    def solve(self, point, rotation, seed):
        q = np.r_[point, rotation.as_rotvec(), seed[6]]
        if np.any(q < self.lower) or np.any(q > self.upper):
            raise ValueError('Test mechanism reach exceeded')
        return q


def test_plans_while_upstream_but_only_starts_inside_region():
    window = interception_window([-.1, 0.], [.028, 0.], [0., .5, -.1, .1], 1., 6., .02)
    assert window['entry'] == pytest.approx(.12/.028)
    assert window['start'] == window['entry']
    assert in_region(window['sync_xy'], [0., .5, -.1, .1], .02)
    assert window['finish'] <= window['exit']


@pytest.mark.parametrize('point,velocity', [
    ([.48, 0.], [.028, 0.]), ([-.1, .2], [.028, 0.]),
    ([.1, 0.], [.028, .08]), ([.1, 0.], [float('nan'), 0.])])
def test_rejects_late_missed_transverse_or_invalid_trajectory(point, velocity):
    with pytest.raises(ValueError):
        interception_window(point, velocity, [0., .5, -.1, .1], 1., 6., .02)


def test_negative_direction_and_stationary_axis_are_handled():
    window = interception_window([.8, .05], [-.04, 0.], [0., .5, -.1, .1], .5, 5.)
    assert window['entry'] == pytest.approx(7.5)
    assert window['sync_xy'][0] == pytest.approx(.5)


def test_complete_duration_not_just_contact_must_fit():
    with pytest.raises(ValueError, match='complete pick'):
        interception_window([.2, 0.], [.04, 0.], [0., .5, -.1, .1], 0., 10.)


@pytest.mark.parametrize('region,margin', [([1, 0, -1, 1], 0), ([0, .1, 0, .1], .06)])
def test_empty_region_is_rejected(region, margin):
    with pytest.raises(ValueError):
        region_box(region, margin)


def test_quintic_and_brake_are_continuous_and_end_stationary():
    assert smooth_segment(.1, .02, 0., 2.5) == pytest.approx((.1, 0.))
    assert smooth_segment(.1, .02, 2.5, 2.5) == pytest.approx((.02, 0.))
    times = np.linspace(0., 2.5, 101)
    samples = np.array([smooth_segment(.1, .02, t, 2.5) for t in times])
    assert np.all(np.diff(samples[:, 0]) <= 1e-12)
    assert np.max(abs(samples[:, 1])) <= 1.875*.08/2.5+1e-12
    delta, velocity = brake_displacement([.028, -.004], .6, .6)
    assert delta == pytest.approx(np.array([.028, -.004])*.3)
    assert velocity == pytest.approx([0., 0.])


def test_tip_velocity_uses_distinct_measured_frames_and_rejects_stale():
    fit = TipVelocity()
    v = np.array([.028, -.004, 0.])
    for t in np.arange(0., .161, .02):
        assert fit.add(t, [.1, .2, .3]+v*t)
        assert not fit.add(t, [100, 100, 100])
    assert fit.estimate(.16) == pytest.approx(v)
    with pytest.raises(ValueError, match='fresh'):
        fit.estimate(.4)


def test_two_pose_samples_cannot_claim_speed_synchronization():
    fit = TipVelocity()
    fit.add(0., [0, 0, 0])
    fit.add(.1, [.003, 0, 0])
    with pytest.raises(ValueError):
        fit.estimate(.1)


def test_measured_speed_and_capture_uncertainty_both_gate_contact():
    args = dict(position=[0., 0.], covariance=np.eye(2)*1e-6,
        object_velocity=[.028, 0.], measured_tip=[0., 0., .1],
        measured_velocity=[.028, 0., 0.], closing_axis=[1., 0.],
        opening_margin=.02, direction=[1., 0.])
    assert matched(**args)
    args['measured_velocity'] = [0., 0., 0.]
    assert not matched(**args)
    args['measured_velocity'] = [.028, 0., 0.]
    args['covariance'] = np.eye(2)*.01**2
    assert not matched(**args)


def test_geometric_jacobian_matches_finite_difference_at_calibrated_tcp(tmp_path):
    axes = ['0 0 1', '0 1 0', '1 0 0', '0 1 0', '1 0 0', '0 0 1', '0 1 0']
    joints = ''.join(
        '<joint name="joint%d"><origin xyz=".03 .02 .07" rpy=".1 -.05 .08"/>'
        '<axis xyz="%s"/><limit lower="-3" upper="3"/></joint>' % (i+1, axis)
        for i, axis in enumerate(axes))
    path = tmp_path/'test.urdf'
    path.write_text('<robot>'+joints+'</robot>')
    kin = Kinematics(path)
    q, tcp, eps = np.linspace(-.3, .4, 7), np.array([.18, .008, -.025]), 1e-6
    numerical = np.zeros((6, 7))
    for i in range(7):
        d = np.zeros(7); d[i] = eps
        plus, minus = kin.fk(q+d), kin.fk(q-d)
        p1 = plus[:3, 3]+plus[:3, :3]@tcp
        p0 = minus[:3, 3]+minus[:3, :3]@tcp
        numerical[:3, i] = (p1-p0)/(2*eps)
        numerical[3:, i] = (Rotation.from_matrix(plus[:3, :3])*
                            Rotation.from_matrix(minus[:3, :3]).inv()).as_rotvec()/(2*eps)
    assert kin.jacobian(q, tcp) == pytest.approx(numerical, abs=1e-7)


def test_closed_loop_matches_moving_object_through_descent_close_and_lift():
    kin, dt = CartesianPlant(), .02
    q = np.r_[0., 0., .12, np.zeros(4)]
    command, qd = q.copy(), np.zeros(7)
    velocity = TipVelocity()
    close_errors, speed_errors = [], []
    previous_qd = qd.copy()
    for index in range(1, 466):
        t = index*dt
        object_v = np.array([.028, .003])
        if t < 1.:
            z, vz = .12, 0.
        elif t < 3.5:
            z, vz = smooth_segment(.12, .022, t-1., 2.5)
        elif t < 4.1:
            z, vz = .022, 0.
        elif t < 6.6:
            z, vz = smooth_segment(.022, .122, t-4.1, 2.5)
        else:
            z, vz = .122, 0.
        if t <= 6.6:
            xy, v = object_v*t, object_v
        else:
            delta, v = brake_displacement(object_v, t-6.6, .6)
            xy = object_v*6.6+delta
        target = np.r_[xy, z]
        command, qd, _ = resolved_rate(kin, q, command, qd, np.zeros(3), target,
            Rotation.identity(), np.r_[v, vz], dt)
        assert np.max(abs(qd)) <= .6+1e-12
        assert np.max(abs(qd-previous_qd)) <= 2.*dt+1e-12
        previous_qd = qd.copy()
        # First-order lag simulates feedback lag instead of copying commands.
        q += (command-q)*dt/(.035+dt)
        velocity.add(t, q[:3])
        if 3.6 <= t <= 4.1:
            actual_v = velocity.estimate(t)
            close_errors.append(np.linalg.norm(q[:2]-xy))
            speed_errors.append(np.linalg.norm(actual_v[:2]-object_v))
            assert matched(xy, np.eye(2)*1e-6, object_v, q[:3], actual_v,
                           [1., 0.], .02, [1., 0.])
    assert max(close_errors) < .002
    assert max(speed_errors) < .002
    assert np.linalg.norm(velocity.estimate(t)) < .008
    assert q[:3] == pytest.approx(target, abs=.002)


@pytest.mark.parametrize('fault', ['stall', 'lag', 'joint_limit', 'singularity'])
def test_servo_rejects_unexecutable_streams(fault):
    kin = CartesianPlant()
    q = np.zeros(7)
    command = q.copy()
    dt = .02
    if fault == 'stall': dt = .1
    if fault == 'lag': command[0] = .1
    if fault == 'joint_limit': kin.upper = np.full(7, -.001)
    if fault == 'singularity': kin.jacobian = lambda *a: np.zeros((6, 7))
    with pytest.raises(ValueError):
        resolved_rate(kin, q, command, np.zeros(7), np.zeros(3), [.001, 0, 0],
                      Rotation.identity(), [.028, 0, 0], dt)


def test_path_preflight_checks_every_flange_and_intermediate_joint_point():
    kin = CartesianPlant()
    seed = np.r_[0., 0., .1, np.zeros(4)]
    ws = dict(x_min=-.2, x_max=.2, y_min=-.2, y_max=.2, z_min=-.05)
    tips = np.array([[0., 0., .1], [.005, 0., .08], [.01, 0., .06]])
    solutions = preflight_follow(kin, seed, np.zeros(3), Rotation.identity(), tips, ws, -.1)
    assert solutions[:, :3] == pytest.approx(tips)
    tips[-1, 0] = .3
    with pytest.raises(ValueError, match='workspace'):
        preflight_follow(kin, seed, np.zeros(3), Rotation.identity(), tips, ws, -.1)
