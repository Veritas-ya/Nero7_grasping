"""Multi-object image tracking with timestamped XY Kalman motion estimates."""
import numpy as np

from .kalman_motion import KalmanMotion

STATUS_TRACKING = 0
STATUS_CONFIRMED = 1
STATUS_LOST = 2


class Track:
    def __init__(self, tid, det, t, speed_cfg, belt_direction=(1.0, 0.0),
                 motion_cfg=None, confirm_frames=2):
        self.id = tid
        self._belt_direction = np.asarray(belt_direction, float)
        config = {key: value for key, value in speed_cfg.items()
                  if key in ('window_s', 'min_samples', 'min_span')}
        config.update(motion_cfg or {})
        config.setdefault('belt_direction', self._belt_direction)
        self.motion = KalmanMotion(np.asarray(det['centroid_base'], float)[:2], t,
                                   **config)
        self._motion_config = config
        self.reacquisitions = 0
        self._reacquire = []
        self._confirm_frames = confirm_frames
        self.hits = 1
        self.misses = 0
        self.confirmed = self.hits >= confirm_frames
        self.observed_centroid = np.asarray(det['centroid_base'], float).copy()
        self.centroid = self.observed_centroid.copy()
        self.velocity = np.zeros(3)
        self._set_observation(det, t)
        self._sync_motion()

    @property
    def status(self):
        if self.misses > 0:
            return STATUS_LOST
        return STATUS_CONFIRMED if self.confirmed else STATUS_TRACKING

    @property
    def motion_covariance(self):
        return self.motion.covariance

    @property
    def motion_process_noise(self):
        return self.motion.process_noise

    @property
    def speed_valid(self):
        return self.motion.speed_valid

    @property
    def speed_ok(self):
        """Compatibility alias; the Kalman estimate is the speed source."""
        return self.speed_valid

    def _set_observation(self, det, t):
        self.u_px = det['u']
        self.v_px = det['v']
        self.observed_centroid = np.asarray(det['centroid_base'], float).copy()
        self.height = det['height']
        self.belt_z = det.get('belt_z', 0.0)
        self.width_px = det['width_px']
        self.area_px = det['area_px']
        self.confidence = det.get('confidence', 1.0)
        self.stamp = float(t)

    def _sync_motion(self):
        self.centroid[:2] = self.motion.state[:2]
        self.centroid[2] = self.observed_centroid[2]
        self.velocity[:2] = self.motion.state[2:]
        self.velocity[2] = 0.0
        self.motion_stamp = self.motion.state_stamp
        self.speed = (float(self.velocity[:2] @ self._belt_direction)
                      if self.speed_valid else None)

    def predict(self, t):
        self.motion.predict(t)
        self._sync_motion()

    def update(self, det, t, dt=None):
        if not self.motion.update(np.asarray(det['centroid_base'], float)[:2], t):
            self._sync_motion()
            return False
        self._set_observation(det, t)
        self._reacquire.clear()
        self.hits += 1
        self.misses = 0
        self.confirmed = self.hits >= self._confirm_frames
        self._sync_motion()
        return True

    def reacquire(self, det, t, samples, max_distance, consistency):
        """Reacquire a unique neighboring object; reset speed qualification."""
        point = np.asarray(det['centroid_base'], float)
        residual = point[:2]-self.centroid[:2]
        if np.linalg.norm(residual) > max_distance:
            self._reacquire.clear()
            return False
        if self._reacquire and (t-self._reacquire[-1][0] > .3
                or np.linalg.norm(residual-self._reacquire[-1][2]) > consistency):
            self._reacquire.clear()
        self._reacquire.append((float(t), det, residual))
        self._reacquire = self._reacquire[-samples:]
        if len(self._reacquire) < samples or np.max(np.linalg.norm(
                np.array([entry[2] for entry in self._reacquire])-residual, axis=1)) > consistency:
            return False
        first_time, first, _ = self._reacquire[0]
        motion = KalmanMotion(np.asarray(first['centroid_base'], float)[:2],
                              first_time, **self._motion_config)
        for stamp, observation, _ in self._reacquire[1:]:
            if not motion.update(np.asarray(observation['centroid_base'], float)[:2], stamp):
                self._reacquire.clear()
                return False
        self.motion = motion
        self._set_observation(det, t)
        self.misses = 0
        self.hits += samples
        self.confirmed = self.hits >= self._confirm_frames
        self.reacquisitions += 1
        self._reacquire.clear()
        self._sync_motion()
        return True

    def mark_missed(self):
        self.misses += 1
        self._sync_motion()


class BlockTracker:
    def __init__(self, belt_model, speed_cfg, max_vel=0.15, gate_radius=0.03,
                 confirm_frames=2, lost_frames=6, motion_cfg=None,
                 lost_timeout=0.5, reacquire_samples=3,
                 reacquire_max_distance=.02, reacquire_consistency=.006):
        self.belt = belt_model
        self.speed_cfg = speed_cfg
        self.motion_cfg = dict(motion_cfg or {})
        self.motion_cfg.setdefault('max_velocity', max_vel)
        self.max_vel = max_vel
        self.gate_radius = gate_radius
        self.confirm_frames = confirm_frames
        self.lost_frames = lost_frames
        self.lost_timeout = lost_timeout
        if (reacquire_samples < 3 or reacquire_max_distance <= 0 or reacquire_consistency <= 0
                or lost_timeout <= 0):
            raise ValueError('Invalid tracking/reacquisition bounds')
        self.reacquire_samples = int(reacquire_samples)
        self.reacquire_max_distance = float(reacquire_max_distance)
        self.reacquire_consistency = float(reacquire_consistency)
        self.tracks = []
        self._next_id = 0
        self._last_t = None

    def update(self, detections, t):
        """Predict, then one-to-one associate gated, real image detections.

        Duplicate/out-of-order timestamps cannot add observations, create
        tracks, change raw coordinates, or age a track out. Lost tracks are
        eligible for re-association during a bounded occlusion interval.
        """
        t = float(t)
        if not np.isfinite(t) or (self._last_t is not None and t <= self._last_t):
            return self.tracks
        self._last_t = t
        self.tracks = [tr for tr in self.tracks
                       if t - tr.stamp <= self.lost_timeout
                       and (tr.confirmed or tr.misses <= self.lost_frames)]
        for tr in self.tracks:
            tr.predict(t)

        valid_detections = []
        for detection in detections:
            position = np.asarray(detection['centroid_base'], float)
            if position.shape == (3,) and np.all(np.isfinite(position)):
                valid_detections.append(detection)

        candidates = []
        neighbors = {}
        for track_index, tr in enumerate(self.tracks):
            # Statistical uncertainty gates association, with a physical
            # limit to prevent a very uncertain ID jumping to another object.
            physical_gate = self.gate_radius + self.max_vel * (t - tr.stamp)
            for detection_index, detection in enumerate(valid_detections):
                innovation = tr.motion.innovation(
                    np.asarray(detection['centroid_base'], float)[:2], t)
                if innovation is None or innovation[1] > physical_gate:
                    continue
                neighbors.setdefault(detection_index, []).append(track_index)
                if innovation[0] <= tr.motion.innovation_gate:
                    candidates.append((innovation[0], innovation[1],
                                       track_index, detection_index))

        matched_tracks, matched_detections = set(), set()
        for _, _, track_index, detection_index in sorted(candidates):
            if track_index in matched_tracks or detection_index in matched_detections:
                continue
            if self.tracks[track_index].update(valid_detections[detection_index], t):
                matched_tracks.add(track_index)
                matched_detections.add(detection_index)

        suppressed = set()
        for detection_index, close_tracks in neighbors.items():
            if detection_index in matched_detections:
                continue
            unmatched = [index for index in close_tracks if index not in matched_tracks]
            if not unmatched:
                continue  # A second object beside a matched track needs its own ID.
            suppressed.add(detection_index)
            if len(close_tracks) != 1 or len(unmatched) != 1:
                continue
            track_index = unmatched[0]
            near_detections = [di for di, indices in neighbors.items() if track_index in indices]
            if len(near_detections) != 1:
                continue
            tr = self.tracks[track_index]
            if tr.reacquire(valid_detections[detection_index], t, self.reacquire_samples,
                            self.reacquire_max_distance, self.reacquire_consistency):
                matched_tracks.add(track_index)
                matched_detections.add(detection_index)

        for track_index, tr in enumerate(self.tracks):
            if track_index not in matched_tracks:
                tr.mark_missed()
            elif tr.misses == 0:
                tr._reacquire.clear()
        for detection_index, detection in enumerate(valid_detections):
            if detection_index in matched_detections:
                continue
            # A statistically rejected jump adjacent to an existing object
            # cannot create a duplicate, newly confirmed ID.
            if detection_index in suppressed:
                continue
            self.tracks.append(Track(self._next_id, detection, t, self.speed_cfg,
                                     self.belt.d_hat, self.motion_cfg,
                                     self.confirm_frames))
            self._next_id += 1

        self.tracks = [tr for tr in self.tracks if tr.confirmed or tr.misses <= self.lost_frames]
        return self.tracks
