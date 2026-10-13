"""Hardware-independent geometry, trajectories and resolved-rate following.

All distances are metres, times seconds and angular quantities radians.
The controller supplies only qualified vision estimates, never belt setpoints.
"""
from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation


def vector(value, size, label):
    result = np.asarray(value, float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError('Invalid '+label)
    return result


def region_box(region, margin=0.):
    box = vector(region, 4, 'follow region').reshape(2, 2).copy()
    if not np.isfinite(margin) or margin < 0:
        raise ValueError('Invalid region margin')
    box[:, 0] += margin
    box[:, 1] -= margin
    if np.any(box[:, 1] <= box[:, 0]):
        raise ValueError('Follow region is empty after margins')
    return box


def in_region(point, region, margin=0.):
    point = vector(point, 2, 'region point')
    box = region_box(region, margin)
    return bool(np.all(point >= box[:, 0]-1e-9) and np.all(point <= box[:, 1]+1e-9))


def interception_window(position, velocity, region, earliest, duration, margin=0.):
    """Find a start that leaves the COMPLETE follow/lift/brake inside the box.

    Uses the actual XY velocity, including transverse drift. Seeing a target
    upstream is enough for planning; execution is admitted only inside the box.
    Returns seconds relative to the position reference time.
    """
    position = vector(position, 2, 'object position')
    velocity = vector(velocity, 2, 'object velocity')
    if not np.isfinite([earliest, duration]).all() or earliest < 0 or duration <= 0:
        raise ValueError('Invalid following time budget')
    box = region_box(region, margin)
    entry, exit_ = 0., float('inf')
    for p, v, limits in zip(position, velocity, box):
        if abs(v) < 1e-9:
            if not limits[0] <= p <= limits[1]:
                raise ValueError('Object trajectory misses the follow region')
            continue
        a, b = sorted((limits-p)/v)
        entry, exit_ = max(entry, a), min(exit_, b)
    start = max(entry, earliest)
    if start+duration > exit_+1e-9 or exit_ < 0:
        raise ValueError('Insufficient region length/time for the complete pick')
    return dict(start=float(start), entry=float(entry), exit=float(exit_),
                finish=float(start+duration), sync_xy=position+velocity*start)


def smooth_segment(start, finish, elapsed, duration):
    """Quintic position with zero velocity/acceleration at both ends."""
    if not np.isfinite([start, finish, elapsed, duration]).all() or duration <= 0:
        raise ValueError('Invalid smooth segment')
    u = float(np.clip(elapsed/duration, 0., 1.))
    s = 10*u**3-15*u**4+6*u**5
    ds = (30*u**2-60*u**3+30*u**4)/duration
    return float(start+(finish-start)*s), float((finish-start)*ds)


def brake_displacement(velocity, elapsed, duration):
    """Integrate a quintic velocity ramp from v to zero; distance = v*T/2."""
    velocity = vector(velocity, 2, 'brake velocity')
    if not np.isfinite([elapsed, duration]).all() or duration <= 0:
        raise ValueError('Invalid brake duration')
    u = float(np.clip(elapsed/duration, 0., 1.))
    s = 10*u**3-15*u**4+6*u**5
    integral = u-2.5*u**4+3*u**5-u**6
    return velocity*duration*integral, velocity*(1-s)


class TipVelocity:
    """Fit ONLY distinct, increasing measured pose samples, not commands."""
    def __init__(self, window=.16, min_span=.06):
        self.window, self.min_span = float(window), float(min_span)
        self.samples = deque(maxlen=80)

    def add(self, stamp, point):
        point = vector(point, 3, 'measured TCP')
        if not np.isfinite(stamp):
            raise ValueError('Invalid feedback timestamp')
        if self.samples and stamp <= self.samples[-1][0]:
            return False
        self.samples.append((float(stamp), point.copy()))
        while self.samples and stamp-self.samples[0][0] > self.window:
            self.samples.popleft()
        return True

    def estimate(self, now):
        if (len(self.samples) < 3 or not 0 <= now-self.samples[-1][0] <= .1
                or self.samples[-1][0]-self.samples[0][0] < self.min_span):
            raise ValueError('Insufficient fresh distinct TCP velocity feedback')
        stamps = np.array([s[0] for s in self.samples])
        points = np.array([s[1] for s in self.samples])
        t = stamps-stamps.mean()
        velocity = t@points/(t@t)
        residual = np.linalg.norm(points-points.mean(axis=0)-t[:, None]*velocity, axis=1).max()
        if residual > .003:
            raise ValueError('Unreliable TCP velocity feedback')
        return velocity


def matched(position, covariance, object_velocity, measured_tip, measured_velocity,
            closing_axis, opening_margin, direction, sigma=3., along_tol=.016,
            cross_tol=.012, velocity_tol=.008):
    from .conveyor_prediction import capture_bounds
    position = vector(position, 2, 'matched position')
    object_velocity = vector(object_velocity, 2, 'matched velocity')
    measured_tip = vector(measured_tip, 3, 'matched TCP')
    measured_velocity = vector(measured_velocity, 3, 'measured velocity')
    closing_axis = vector(closing_axis, 2, 'closing axis')
    length = np.linalg.norm(closing_axis)
    if length < .5 or not np.isfinite(opening_margin) or opening_margin <= 0:
        return False
    bounds = capture_bounds(position, covariance, measured_tip[:2], direction, sigma)
    axis = closing_axis/length
    closing_bound = abs((position-measured_tip[:2])@axis)+sigma*np.sqrt(
        max(0., axis@covariance@axis))
    return bool(bounds['along_bound'] <= along_tol and bounds['cross_bound'] <= cross_tol
                and closing_bound <= opening_margin
                and np.linalg.norm(measured_velocity[:2]-object_velocity) <= velocity_tol)


def resolved_rate(kin, joints, command, previous_velocity, tcp, target, rotation,
                  feedforward, dt, gain=4., joint_speed=.6, joint_accel=2.,
                  linear_speed=.10, angular_speed=.3, damping=.015):
    """Damped geometric IK; position and velocity feedback precede admission.

    This creates a smooth stream of positions for AGX move_js; the driver does
    not consume a Cartesian velocity. Reject poor conditioning/unreachable
    twist rather than accepting an arbitrary slowed conveyor-follow command.
    """
    q = vector(joints, 7, 'actual joints')
    command = vector(command, 7, 'joint command')
    previous_velocity = vector(previous_velocity, 7, 'previous joint velocity')
    tcp = vector(tcp, 3, 'TCP offset')
    target = vector(target, 3, 'TCP reference')
    feedforward = vector(feedforward, 3, 'TCP feedforward')
    scalars = [dt, gain, joint_speed, joint_accel, linear_speed, angular_speed, damping]
    if not np.isfinite(scalars).all() or min(scalars) <= 0 or dt > .06:
        raise ValueError('Invalid/stalled follow control interval')
    if np.max(np.abs(command-q)) > .08:
        raise ValueError('Actual joints lag the stream by more than 0.08 rad')
    transform = kin.fk(q)
    actual_rotation = Rotation.from_matrix(transform[:3, :3])
    tip = transform[:3, 3]+actual_rotation.apply(tcp)
    error = target-tip
    angular_error = (rotation*actual_rotation.inv()).as_rotvec()
    if np.linalg.norm(error) > .04 or np.linalg.norm(angular_error) > .12:
        raise ValueError('TCP departed from the verified follow trajectory')
    linear = feedforward+gain*error
    if np.linalg.norm(linear) > linear_speed:
        linear *= linear_speed/np.linalg.norm(linear)
    angular = gain*angular_error
    if np.linalg.norm(angular) > angular_speed:
        angular *= angular_speed/np.linalg.norm(angular)
    jacobian = kin.jacobian(q, tcp)
    weighted = jacobian.copy()
    weighted[3:] *= .25
    desired = np.r_[linear, angular*.25]
    singular = np.linalg.svd(weighted, compute_uv=False)
    if singular[-1] < .003:
        raise ValueError('Follow trajectory approaches a kinematic singularity')
    desired_qd = weighted.T@np.linalg.solve(
        weighted@weighted.T+np.eye(6)*damping**2, desired)
    factor = max(1., np.max(np.abs(desired_qd))/joint_speed)
    desired_qd /= factor
    change = desired_qd-previous_velocity
    factor = max(1., np.max(np.abs(change))/(joint_accel*dt))
    qd = previous_velocity+change/factor
    next_command = command+qd*dt
    if np.any(next_command < kin.lower) or np.any(next_command > kin.upper):
        raise ValueError('Follow stream reaches a joint limit')
    if np.linalg.norm((jacobian@qd)[:3]-linear) > .03:
        raise ValueError('Joint limits prevent the requested following speed')
    return next_command, qd, tip


def preflight_follow(kin, seed, tcp, rotation, tips, workspace, belt_z,
                     joint_speed=.6, sample_dt=.25):
    """Sample IK plus midpoints of joint interpolation before streaming."""
    seed = vector(seed, 7, 'path seed')
    tcp = vector(tcp, 3, 'path TCP')
    tips = np.asarray(tips, float)
    if tips.ndim != 2 or tips.shape[1] != 3 or not np.isfinite(tips).all() or len(tips) < 2:
        raise ValueError('Invalid follow preflight points')
    previous, solutions = seed.copy(), []
    for index, tip in enumerate(tips):
        flange = tip-rotation.apply(tcp)
        if not (workspace['x_min'] <= flange[0] <= workspace['x_max']
                and workspace['y_min'] <= flange[1] <= workspace['y_max']
                and flange[2] >= workspace['z_min']):
            raise ValueError('Following flange outside workspace')
        q = kin.solve(flange, rotation, previous)
        if index and np.max(np.abs(q-previous))/sample_dt > joint_speed:
            raise ValueError('Follow trajectory exceeds joint speed limit')
        for alpha in np.linspace(0., 1., 5):
            intermediate = previous*(1-alpha)+q*alpha
            transform = kin.fk(intermediate)
            point = transform[:3, 3]+transform[:3, :3]@tcp
            if index and (point[2] < belt_z+.005
                    or np.linalg.norm(point-(tips[index-1]*(1-alpha)+tip*alpha)) > .01):
                raise ValueError('Joint interpolation leaves the clear Cartesian path')
            if index and np.linalg.svd(kin.jacobian(intermediate, tcp)*np.array(
                    [1., 1., 1., .25, .25, .25])[:, None], compute_uv=False)[-1] < .003:
                raise ValueError('Preflight encountered a singularity')
        previous = q
        solutions.append(q.copy())
    return np.asarray(solutions)
