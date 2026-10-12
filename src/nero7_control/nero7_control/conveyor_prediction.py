"""Kalman forecasts and fixed-line interception timing; legacy fit helpers retained."""
import numpy as np


def fit_velocity(samples, direction, window_s=1.2):
    """Samples are (image_stamp, base_xy); reject noisy/cross-belt motion."""
    if not samples:
        raise ValueError('No visual observations')
    end = samples[-1][0]
    recent = [s for s in samples if end-s[0] <= window_s]
    times = np.array([s[0] for s in recent], float)
    xy = np.array([s[1] for s in recent], float)
    if len(times) < 16 or times[-1]-times[0] < .6 or np.any(np.diff(times) <= 0):
        raise ValueError('Insufficient distinct visual observations')
    times -= times[0]
    slope, offset = np.polyfit(times, xy, 1)
    residual = np.sqrt(np.mean(np.sum((xy-(times[:, None]*slope+offset))**2, axis=1)))
    direction = np.asarray(direction, float)
    direction /= np.linalg.norm(direction)
    speed = float(slope@direction)
    if not .008 <= speed <= .04 or residual > .006 or np.linalg.norm(slope-direction*speed) > .012:
        raise ValueError('Visual velocity outside trial limits or unreliable')
    return direction*speed, float(residual)


def contact_prediction(position, image_stamp, now, velocity, horizon, offset):
    age = now-image_stamp
    if not 0 <= age <= .2 or horizon < 0:
        raise ValueError('Stale image or invalid contact horizon')
    return np.asarray(position)+np.asarray(offset)+np.asarray(velocity)*(age+horizon)


def contact_delay(close_seconds, open_width, calibration_goal, raw_width):
    if not calibration_goal < open_width or not 0 < raw_width < open_width:
        raise ValueError('Invalid gripper timing widths')
    # Contact occurs before an empty close reaches the smaller commanded width.
    return float(np.clip(close_seconds*(open_width-raw_width)/(open_width-calibration_goal), .15, 2.))


def _covariance(value, size):
    matrix = np.asarray(value, float)
    if matrix.shape != (size, size) or not np.isfinite(matrix).all():
        raise ValueError('Invalid covariance shape or values')
    if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=1e-6):
        raise ValueError('Covariance is not symmetric')
    if np.linalg.eigvalsh(matrix).min() < -1e-12:
        raise ValueError('Covariance is not positive semidefinite')
    return (matrix+matrix.T)/2


def forecast_motion(state, covariance, dt, process_noise):
    """Propagate [x,y,vx,vy] using the perception filter's exact noise model."""
    state = np.asarray(state, float)
    if state.shape != (4,) or not np.isfinite(state).all() or not np.isfinite(dt) or dt < 0:
        raise ValueError('Invalid motion state or forecast time')
    covariance = _covariance(covariance, 4)
    density = _covariance(process_noise, 2)
    transition = np.eye(4)
    transition[:2, 2:] = np.eye(2)*dt
    noise = np.block([[density*dt**3/3, density*dt**2/2],
                      [density*dt**2/2, density*dt]])
    predicted = transition@covariance@transition.T+noise
    return transition@state, (predicted+predicted.T)/2


def motion_at_contact(track, now, horizon, offset, offset_covariance, timing_std=.15, allow_lost=False):
    """Forecast a grasp point, including observation age and timing uncertainty.

    ``stamp`` is the last accepted observation; ``motion_stamp`` is the state
    reference time. Short prediction through loss is allowed only after start.
    """
    values = [now, horizon, timing_std, track.stamp, track.motion_stamp]
    if not np.isfinite(values).all() or horizon < 0 or timing_std < 0:
        raise ValueError('Invalid contact prediction timing')
    age, state_age = now-track.stamp, now-track.motion_stamp
    if track.motion_stamp+1e-6 < track.stamp:
        raise ValueError('Motion state predates its latest observation')
    if not 0 <= age <= .2 or not 0 <= state_age <= .2:
        raise ValueError('Stale or future visual motion estimate')
    if track.status != 1 and not (allow_lost and track.status == 2 and age <= .15):
        raise ValueError('Target not confirmed or fresh enough to trigger')
    if not track.speed_valid:
        raise ValueError('Kalman velocity has not converged')
    offset = np.asarray(offset, float)
    velocity = np.asarray(track.velocity, float)[:2]
    if offset.shape != (2,) or velocity.shape != (2,) or not np.isfinite(offset).all():
        raise ValueError('Invalid object-relative grasp offset')
    state = np.r_[track.centroid_x, track.centroid_y, velocity]
    predicted, covariance = forecast_motion(state, track.motion_covariance, state_age+horizon,
                                             track.motion_process_noise)
    position_covariance = covariance[:2, :2]+_covariance(offset_covariance, 2)
    position_covariance += np.outer(velocity, velocity)*timing_std**2
    return dict(position=predicted[:2]+offset, covariance=position_covariance,
                velocity=predicted[2:], age=float(age), state_age=float(state_age))


def capture_bounds(position, covariance, goal, direction, sigma=3.):
    """Mean error plus confidence radius along/across the moving belt."""
    direction = np.asarray(direction, float)
    position, goal = np.asarray(position, float), np.asarray(goal, float)
    if (direction.shape != (2,) or position.shape != (2,) or goal.shape != (2,)
            or not np.isfinite(np.r_[position, goal, direction, sigma]).all()
            or sigma < 0 or np.linalg.norm(direction) < 1e-9):
        raise ValueError('Invalid capture geometry')
    direction = direction/np.linalg.norm(direction)
    normal = np.array([-direction[1], direction[0]])
    covariance = _covariance(covariance, 2)
    error = position-goal
    along, cross = float(error@direction), float(error@normal)
    return dict(along_error=along, cross_error=cross,
                along_bound=abs(along)+sigma*np.sqrt(max(0., direction@covariance@direction)),
                cross_bound=abs(cross)+sigma*np.sqrt(max(0., normal@covariance@normal)))


def stage_window(remaining, approach, descend, contact, margin):
    """Upstream startup range derived from the complete remaining motion time."""
    if not np.isfinite([remaining, approach, descend, contact, margin]).all() or min(
            remaining, approach, descend, contact, margin) < 0:
        raise ValueError('Invalid stage timing')
    duration = approach+descend+contact
    if remaining > duration+margin+1e-9:
        return 'WAIT'
    if remaining+1e-9 < duration:
        return 'LATE'
    return 'START'
