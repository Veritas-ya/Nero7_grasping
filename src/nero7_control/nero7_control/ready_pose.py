"""Model checks for the named waiting pose; no ROS or hardware commands."""
import numpy as np


JOINT_NAMES = ['joint%d' % i for i in range(1, 8)]


def joint_vector(values):
    q = np.asarray(values, float)
    if q.shape != (7,) or not np.isfinite(q).all():
        raise ValueError('Expected seven finite joint angles in radians')
    return q


def ordered_joints(names, positions):
    if len(names) != len(positions) or any(names.count(n) != 1 for n in JOINT_NAMES):
        raise ValueError('Feedback must name every arm joint exactly once')
    return joint_vector([positions[names.index(n)] for n in JOINT_NAMES])


def require_limits(kin, values):
    q = joint_vector(values)
    outside = np.flatnonzero((q < kin.lower) | (q > kin.upper))
    if outside.size:
        detail = '; '.join('joint%d=%.2f deg (%.2f..%.2f)' % (
            i+1, np.rad2deg(q[i]), np.rad2deg(kin.lower[i]), np.rad2deg(kin.upper[i]))
            for i in outside)
        raise ValueError('Joint limits: '+detail)
    return q


def checked_point(kin, q, tcp, belt_z, profile):
    q = require_limits(kin, q)
    transform = kin.fk(q)
    flange = transform[:3, 3]
    tip = flange+transform[:3, :3]@tcp
    ws = profile['workspace']
    if not (ws['x_min'] <= flange[0] <= ws['x_max']
            and ws['y_min'] <= flange[1] <= ws['y_max'] and flange[2] >= ws['z_min']):
        raise ValueError('Flange outside waiting-pose workspace')
    if tip[2] < belt_z+profile['min_tip_clearance_m']:
        raise ValueError('Fingertip clearance below waiting-pose minimum')
    return transform, tip


def plan_ready(kin, seed, profile, tcp, belt_z):
    if profile.get('joint_names') != JOINT_NAMES or profile.get('frame_id') != 'base_link':
        raise ValueError('Waiting pose must use joint1..joint7 and base_link')
    seed, goal = require_limits(kin, seed), require_limits(kin, profile['joints_rad'])
    tcp = np.asarray(tcp, float)
    if tcp.shape != (3,) or not np.isfinite(tcp).all() or not np.isfinite(belt_z):
        raise ValueError('Invalid calibrated TCP or belt height')
    limits = [profile['max_joint_step_rad'], profile['min_tip_clearance_m']]
    if not np.isfinite(limits).all() or not 0 < limits[0] <= 1.2 or limits[1] < .05:
        raise ValueError('Waiting-pose step/clearance limits cannot be relaxed')
    step = float(np.max(np.abs(goal-seed)))
    if step > profile['max_joint_step_rad']:
        raise ValueError('Joint step %.3f rad exceeds waiting-pose limit' % step)
    minimum = float('inf')
    for alpha in np.linspace(0, 1, 150):
        _, tip = checked_point(kin, seed*(1-alpha)+goal*alpha, tcp, belt_z, profile)
        minimum = min(minimum, float(tip[2]-belt_z))
    transform, tip = checked_point(kin, goal, tcp, belt_z, profile)
    return dict(name=profile.get('name', 'grasp_ready'), seed_joints_rad=seed.tolist(),
                target_joints_rad=goal.tolist(), target_joints_deg=np.rad2deg(goal).tolist(),
                target_flange_m=transform[:3, 3].tolist(), target_tip_m=tip.tolist(),
                min_sampled_tip_clearance_m=minimum, max_joint_step_rad=step,
                path_check='150 samples of nominal joint interpolation; no full collision model')
