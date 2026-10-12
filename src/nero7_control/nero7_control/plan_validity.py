"""Pure checks shared by preview and execution of stationary plans."""
import numpy as np


def candidate_matches(result, track, now, max_age=.3):
    if result is None or result.get('status') != 0 or result.get('target_id') != track.id:
        return False
    stamps = np.asarray([now, result.get('stamp', np.nan), track.stamp], float)
    return bool(np.isfinite(stamps).all() and 0 <= now-track.stamp <= .2
                and 0 <= now-result['stamp'] <= max_age
                and abs(result['stamp']-track.stamp) <= .1)


def stationary_plan_current(plan, track, result, now, joints, position, rotation,
                            max_age=.5, max_move=.008):
    if plan is None or not candidate_matches(result, track, now):
        return False
    if plan.get('track_id') != track.id or not 0 <= now-plan['image_stamp'] <= max_age:
        return False
    center = np.array([track.centroid_x, track.centroid_y, track.centroid_z])
    if not np.isfinite(center).all() or np.linalg.norm(center-plan['track_center']) > max_move:
        return False
    yaw_delta = (result['yaw']-plan['network_yaw']+np.pi/2) % np.pi-np.pi/2
    if (abs(yaw_delta) > .1 or abs(float(result['raw_width_mm'])/1000-plan['raw_width']) > .004
            or np.linalg.norm(np.asarray(result['center'])-plan['network_center']) > max_move):
        return False
    return seed_matches(plan, joints, position, rotation)


def seed_matches(plan, joints, position, rotation):
    if joints is None or len(joints) < 7:
        return False
    seed = np.asarray(joints[:7], float)
    position = np.asarray(position, float)
    return bool(np.isfinite(seed).all() and np.isfinite(position).all()
                and np.max(np.abs(seed-plan['seed_joints'])) < .03
                and np.linalg.norm(position-plan['seed_flange']) < .005
                and (rotation.inv()*plan['seed_rotation']).magnitude() < .05)
