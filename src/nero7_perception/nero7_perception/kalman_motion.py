"""Timestamped XY constant-velocity Kalman filter, independent of ROS.

State order is ``[x, y, vx, vy]`` in metres and seconds. Acceleration noise
parameters are square roots of continuous white-acceleration spectral
density (m/s**1.5). An interception forecast must include that process noise.
"""
from collections import deque

import numpy as np


def transition_and_noise(dt, acceleration_noise):
    """Return CV transition and white-acceleration covariance.

    ``acceleration_noise`` is either an isotropic spectral-density standard
    deviation or a full 2x2 acceleration spectral-density covariance.
    """
    dt = float(dt)
    if not np.isfinite(dt) or dt < 0:
        raise ValueError('prediction interval must be finite and nonnegative')
    transition = np.eye(4)
    transition[0, 2] = transition[1, 3] = dt
    density = np.asarray(acceleration_noise, dtype=float)
    if density.ndim == 0:
        density = np.eye(2) * float(density) ** 2
    if (density.shape != (2, 2) or not np.all(np.isfinite(density))
            or not np.allclose(density, density.T)
            or np.linalg.eigvalsh(density)[0] < -1e-16):
        raise ValueError('acceleration spectral density must be positive semidefinite')
    noise = np.block([[density * dt ** 3 / 3.0, density * dt ** 2 / 2.0],
                      [density * dt ** 2 / 2.0, density * dt]])
    return transition, noise


class KalmanMotion:
    """Accept distinct, innovation-gated positions and retain observation age.

    ``state_stamp`` is the filtered/predicted state's reference time;
    ``observation_stamp`` only changes for accepted image observations.
    Prediction never counts as another measurement. Qualification uses a
    recent window of accepted measurements and velocity covariance.
    """

    def __init__(self, xy, stamp, measurement_std=0.002,
                 accel_noise_std=0.002, initial_velocity_std=0.10,
                 innovation_gate=9.210340371976184,
                 min_samples=8, min_span=0.3, window_s=1.2,
                 max_velocity_std=0.010, observation_max_age=0.15,
                 qualification_gap=0.3, max_velocity=0.15,
                 belt_direction=None, cross_accel_noise_std=0.0005):
        xy = np.asarray(xy, dtype=float)
        if xy.shape != (2,) or not np.all(np.isfinite(xy)) or not np.isfinite(stamp):
            raise ValueError('initial XY position and timestamp must be finite')
        positive = (measurement_std, initial_velocity_std, innovation_gate,
                    min_span, window_s, max_velocity_std, observation_max_age,
                    qualification_gap, max_velocity)
        if (not all(np.isfinite(value) and value > 0 for value in positive)
                or not np.isfinite(accel_noise_std) or accel_noise_std < 0
                or not np.isfinite(cross_accel_noise_std) or cross_accel_noise_std < 0
                or min_samples < 2 or window_s < min_span):
            raise ValueError('invalid Kalman noise or qualification parameters')
        self.measurement_std = float(measurement_std)
        self.accel_noise_std = float(accel_noise_std)
        self.cross_accel_noise_std = float(cross_accel_noise_std)
        self.process_noise = np.eye(2) * accel_noise_std ** 2
        if belt_direction is not None:
            direction = np.asarray(belt_direction, dtype=float)
            norm = np.linalg.norm(direction)
            if direction.shape != (2,) or not np.all(np.isfinite(direction)) or norm <= 0:
                raise ValueError('belt direction must be a finite nonzero XY vector')
            direction = direction / norm
            cross_direction = np.array([-direction[1], direction[0]])
            self.process_noise = (accel_noise_std ** 2 * np.outer(direction, direction) +
                                  cross_accel_noise_std ** 2 *
                                  np.outer(cross_direction, cross_direction))
        self.innovation_gate = float(innovation_gate)
        self.min_samples = int(min_samples)
        self.min_span = float(min_span)
        self.window_s = float(window_s)
        self.max_velocity_std = float(max_velocity_std)
        self.observation_max_age = float(observation_max_age)
        self.qualification_gap = float(qualification_gap)
        self.max_velocity = float(max_velocity)
        self.state = np.r_[xy, np.zeros(2)]
        self.covariance = np.diag([measurement_std ** 2] * 2 +
                                  [initial_velocity_std ** 2] * 2)
        self.state_stamp = float(stamp)
        self.observation_stamp = float(stamp)
        self.observed_xy = xy.copy()
        self.accepted_stamps = deque([float(stamp)])
        self.accepted_count = 1
        self.rejected_count = 0
        self._measurement_covariance = np.eye(2) * measurement_std ** 2

    def forecast(self, stamp):
        """Return state/covariance at an absolute timestamp without mutation."""
        transition, noise = transition_and_noise(
            float(stamp) - self.state_stamp, self.process_noise)
        state = transition @ self.state
        covariance = transition @ self.covariance @ transition.T + noise
        return state, (covariance + covariance.T) * 0.5

    def predict(self, stamp):
        stamp = float(stamp)
        if not np.isfinite(stamp) or stamp < self.state_stamp:
            return False
        if stamp == self.state_stamp:
            return True
        self.state, self.covariance = self.forecast(stamp)
        self.state_stamp = stamp
        self._trim_samples()
        return True

    def innovation(self, xy, stamp):
        """Return (Mahalanobis squared distance, Euclidean distance), or None."""
        xy = np.asarray(xy, dtype=float)
        if (xy.shape != (2,) or not np.all(np.isfinite(xy))
                or not np.isfinite(stamp) or stamp < self.state_stamp):
            return None
        state, covariance = self.forecast(stamp)
        residual = xy - state[:2]
        innovation_covariance = covariance[:2, :2] + self._measurement_covariance
        distance_sq = float(residual @ np.linalg.solve(innovation_covariance, residual))
        return distance_sq, float(np.linalg.norm(residual))

    def update(self, xy, stamp):
        """Return True only when a new, finite image position was accepted."""
        stamp = float(stamp)
        if (not np.isfinite(stamp) or stamp <= self.observation_stamp
                or stamp < self.state_stamp):
            return False
        innovation = self.innovation(xy, stamp)
        if innovation is None:
            self.rejected_count += 1
            return False
        self.predict(stamp)
        if innovation[0] > self.innovation_gate:
            self.rejected_count += 1
            return False
        xy = np.asarray(xy, dtype=float)
        innovation_covariance = self.covariance[:2, :2] + self._measurement_covariance
        gain = np.linalg.solve(innovation_covariance, self.covariance[:2, :]).T
        self.state += gain @ (xy - self.state[:2])
        # Joseph form preserves positive covariance under finite precision.
        correction = np.eye(4)
        correction[:, :2] -= gain
        self.covariance = (correction @ self.covariance @ correction.T +
                           gain @ self._measurement_covariance @ gain.T)
        self.covariance = (self.covariance + self.covariance.T) * 0.5
        if stamp - self.observation_stamp > self.qualification_gap:
            self.accepted_stamps.clear()
        self.observation_stamp = stamp
        self.observed_xy = xy.copy()
        self.accepted_stamps.append(stamp)
        self.accepted_count += 1
        self._trim_samples()
        return True

    def _trim_samples(self):
        cutoff = self.state_stamp - self.window_s
        while self.accepted_stamps and self.accepted_stamps[0] < cutoff:
            self.accepted_stamps.popleft()

    @property
    def qualification_reason(self):
        if len(self.accepted_stamps) < self.min_samples:
            return 'insufficient_observations'
        if self.accepted_stamps[-1]-self.accepted_stamps[0] < self.min_span:
            return 'insufficient_time_span'
        if self.state_stamp-self.observation_stamp > self.observation_max_age:
            return 'observation_stale'
        if np.linalg.eigvalsh(self.covariance[2:, 2:])[-1] > self.max_velocity_std**2:
            return 'velocity_uncertain'
        if np.linalg.norm(self.state[2:]) > self.max_velocity:
            return 'velocity_out_of_range'
        return 'qualified'

    @property
    def speed_valid(self):
        return self.qualification_reason == 'qualified'
