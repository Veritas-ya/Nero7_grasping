"""One fixed-line pick using live Kalman state and a timed startup region.

Plan upstream without moving. Start approach only inside the time budget's
startup region; descend only when the contact confidence bounds fit the grasp.
The default execution_disabled path performs preview planning without commands.
"""
import json
import time
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import yaml
from std_srvs.srv import SetBool
from rcl_interfaces.srv import GetParameters
from scipy.spatial.transform import Rotation
from .static_once import StaticOnce
from .kinematics import yaw_orientations
from .conveyor_prediction import contact_delay, motion_at_contact, capture_bounds, stage_window


class ConveyorOnce(StaticOnce):
    ACTIVE_STATES = {
        'CONVEYOR_CALIB_ENABLE', 'CONVEYOR_CALIB_OPEN', 'CONVEYOR_CALIB_APPROACH',
        'CONVEYOR_CALIB_DESCEND', 'CONVEYOR_CALIB_CLOSE', 'CONVEYOR_CALIB_REOPEN',
        'CONVEYOR_CALIB_RAISE', 'CONVEYOR_APPROACH', 'CONVEYOR_DESCEND',
        'CONVEYOR_START_ENABLE', 'CONVEYOR_START_OPEN', 'CONVEYOR_WAIT_GATE',
        'CONVEYOR_CLOSE', 'CONVEYOR_LIFT', 'CONVEYOR_HOLD', 'CONVEYOR_RETURN',
        'CONVEYOR_RELEASE', 'CONVEYOR_RETRACT', 'CONVEYOR_RECOVER_OPEN', 'CONVEYOR_RECOVER_RAISE',
    }

    def __init__(self, node):
        super().__init__(node)
        declare = lambda name, value: node.declare_parameter(name, value).value
        self.plan_file = declare('conveyor_plan_file', '/tmp/nero7_conveyor_plan.json')
        self.timing_file = declare('conveyor_timing_file', '/tmp/nero7_conveyor_timings.json')
        with open(node._p['belt_file']) as f:
            self.belt_data = yaml.safe_load(f)
        self.belt, self.belt_z = node._belt, float(self.belt_data['belt_height'])
        self.approach_seconds = float(declare('conveyor_approach_seconds', 2.))
        self.prepare_seconds = float(declare('conveyor_prepare_seconds', 1.5))
        self.planning_reserve = float(declare('conveyor_planning_reserve', 2.5))
        self.start_margin = float(declare('conveyor_start_margin', .8))
        self.trigger_margin = float(declare('conveyor_trigger_margin', .12))
        self.timing_std = float(declare('conveyor_timing_std', .15))
        self.sigma = float(declare('conveyor_confidence_sigma', 3.))
        self.along_tolerance = float(declare('conveyor_along_tolerance', .025))
        self.cross_tolerance = float(declare('conveyor_cross_tolerance', .016))
        self.stable_frames = int(declare('conveyor_stable_frames', 3))
        self.timing_max_age = float(declare('conveyor_timing_max_age', 3600.))
        self.calibration_only = declare('conveyor_calibration_only', False)
        self.calibration_yaw = float(declare('conveyor_calibration_yaw', 1.444185508449333))
        self.state = 'CONVEYOR_CALIB_READY' if self.calibration_only else 'CONVEYOR_WAIT_TARGET'
        self.calibration_goal, self.stroke = .0205, .078
        # Estimates support preview. Execution requires a measured timing record.
        self.descend_seconds, self.close_seconds = 3., .3
        self.timing_verified = False
        self.calibration_rotation = None
        self.history, self.candidates = {}, {}
        self.acquisition_id, self.acquisition_started = None, 0.
        self.rejected_until = {}
        self.calibration_empty_since = None
        self.offset_xy = np.zeros(2)
        self.offset_covariance = np.eye(2)*.003**2
        self.stable_count, self.stable_stamp = 0, None
        self.report = dict(mode='conveyor_once', estimator='Kalman XY CV', belt=self.belt_data,
                           events=[], predictions=[], timing_verified=False,
                           time_source='ros_sim' if self.use_sim_time else 'wall')
        self.planner = ThreadPoolExecutor(max_workers=1)
        self.planning_future = None
        self.calibration_future = None
        driver_node = declare('conveyor_driver_node', '/agx_arm_ctrl_single_node')
        self.profile_client = node.create_client(GetParameters, driver_node.rstrip('/')+'/get_parameters')
        self.profile_future = None
        self.driver_speed = None
        self.profile_checked = 0.
        self.loaded_timings = declare('conveyor_load_timings_file', '')
        self.previous_timing = None
        if self.loaded_timings:
            with open(self.loaded_timings) as f:
                previous = json.load(f)
            if previous['belt'] != self.belt_data:
                raise ValueError('Timing record does not match current belt geometry')
            self.descend_seconds = float(previous['descend_seconds'])
            self.close_seconds = float(previous['empty_close_seconds'])
            if not .4 < self.descend_seconds < 8 or not .15 < self.close_seconds < 4:
                raise ValueError('Invalid loaded motion timing')
            self.previous_timing = previous
            self.report['timing_source'] = self.loaded_timings
        scalars = [self.approach_seconds, self.prepare_seconds, self.planning_reserve, self.start_margin,
                   self.trigger_margin, self.timing_std, self.sigma, self.along_tolerance,
                   self.cross_tolerance, self.timing_max_age]
        if not np.isfinite(scalars).all() or min(scalars) <= 0 or self.stable_frames < 2:
            raise ValueError('Invalid conveyor scheduling parameters')
        if not np.isfinite(self.calibration_yaw):
            raise ValueError('Invalid calibration closing direction')

    def shutdown(self):
        super().shutdown()
        self.planner.shutdown(wait=True, cancel_futures=True)

    def _set(self, state):
        super()._set(state.replace('STATIC_', 'CONVEYOR_'))
        self.stable_count, self.stable_stamp = 0, None
        self.report['events'].append(dict(state=self.state, at_unix=time.time()))
        self._save()

    def _save(self):
        with open(self.timing_file, 'w') as f:
            json.dump(self.report, f, indent=2)

    def _execution_enabled(self):
        return bool(self.node.get_parameter('execution_enabled').value) and not self.node._p['dry_run']

    def _skip(self, reason):
        self.node.get_logger().warning('Moving target skipped: '+reason)
        self.report['last_skip'] = reason
        if self.track is not None:
            self.rejected_until[self.track.id] = time.monotonic()+3.
        self.acquisition_id = None
        self.plan, self.track = None, None
        self._set('CONVEYOR_WAIT_TARGET')

    def _driver_profile(self):
        now = time.monotonic()
        if self.profile_future is not None and self.profile_future.done():
            response = self.profile_future.result()
            self.profile_future = None
            self.driver_speed = int(response.values[0].integer_value) if response and response.values else None
            self.profile_checked = now
        if (self.profile_future is None and now-self.profile_checked > 2
                and self.profile_client.service_is_ready()):
            request = GetParameters.Request(); request.names = ['speed_percent']
            self.profile_future = self.profile_client.call_async(request)
        return self.driver_speed is not None and 0 < self.driver_speed <= 100 and now-self.profile_checked < 5

    def _observe(self, now):
        for tr in self.node._tracks:
            if tr.status != 1 or not 0 <= now-tr.stamp <= .2:
                continue
            samples = self.history.setdefault(tr.id, [])
            if not samples or tr.stamp > samples[-1][0]:
                samples.append((tr.stamp, tr.observed_position.copy()))
            self.history[tr.id] = [s for s in samples if tr.stamp-s[0] < 3]
        for tid in list(self.history):
            if not self.history[tid] or now-self.history[tid][-1][0] > 1:
                self.history.pop(tid); self.candidates.pop(tid, None)

    def _estimate(self, tr, now, horizon=0., allow_lost=False, timing_std=None,
                  offset=None, offset_covariance=None):
        estimate = motion_at_contact(tr, now, horizon,
                                     self.offset_xy if offset is None else offset,
                                     self.offset_covariance if offset_covariance is None else offset_covariance,
                                     self.timing_std if timing_std is None else timing_std, allow_lost)
        velocity = estimate['velocity']
        speed = float(velocity@self.belt.d_hat)
        if not .008 <= speed <= .04 or np.linalg.norm(velocity-self.belt.d_hat*speed) > .012:
            raise ValueError('Filtered motion outside conveyor trial limits')
        return estimate

    def _target(self, now, allow_lost=False):
        tr = next((t for t in self.node._tracks if self.track is not None and t.id == self.track.id), None)
        if tr is None:
            raise ValueError('Moving target missing')
        self._estimate(tr, now, allow_lost=allow_lost)
        return tr

    def _remaining(self, tr, now, offset=None, offset_covariance=None):
        estimate = self._estimate(tr, now, timing_std=0., offset=offset, offset_covariance=offset_covariance)
        return float((self.belt.gate_s-self.belt.s_of(estimate['position'])) /
                     (estimate['velocity']@self.belt.d_hat))

    def _prediction(self, tr, now, horizon, allow_lost=False):
        estimate = self._estimate(tr, now, horizon, allow_lost)
        bounds = capture_bounds(estimate['position'], estimate['covariance'], self.plan['grasp'][:2],
                                self.belt.d_hat, self.sigma)
        self.report['predictions'].append(dict(
            at_unix=time.time(), at_ros_time=now, state=self.state, observation_age_s=estimate['age'],
            velocity_m_s=estimate['velocity'].tolist(), mean_xy_m=estimate['position'].tolist(),
            covariance_xy_m2=estimate['covariance'].tolist(), **{k: float(v) for k, v in bounds.items()}))
        self.report['predictions'] = self.report['predictions'][-300:]
        return estimate, bounds

    def _capture_ok(self, estimate, bounds):
        if bounds['along_bound'] > self.along_tolerance or bounds['cross_bound'] > self.cross_tolerance:
            return False
        closing_axis = self.plan['rotation'].apply(self.axis)[:2]
        closing_axis /= np.linalg.norm(closing_axis)
        error = estimate['position']-self.plan['grasp'][:2]
        closing_bound = abs(error@closing_axis)+self.sigma*np.sqrt(
            max(0., closing_axis@estimate['covariance']@closing_axis))
        opening_margin = (self.open_width-self.plan['raw_width'])/2-.002
        return closing_bound <= opening_margin

    def _stable(self, tr):
        if tr.stamp != self.stable_stamp:
            self.stable_stamp = tr.stamp
            self.stable_count += 1
        return self.stable_count >= self.stable_frames

    def _calibration_plan(self):
        xy = self.belt.point_of(self.belt.gate_s)
        center = np.r_[xy, self.belt_z+.062]
        track = SimpleNamespace(centroid_x=xy[0], centroid_y=xy[1], belt_z=self.belt_z, id=-1)
        result = dict(center=center, yaw=self.calibration_yaw, raw_width_mm=24.,
                      width_mm=self.calibration_goal*1000, score=0.)
        self.plan = self._make_plan(result, track, approach_height=.06, source='non_contact_timing_calibration')
        if abs(self.plan['above'][2]-self.plan['grasp'][2]-self.stroke) > .002:
            raise ValueError('Calibration stroke does not match configured grasp geometry')
        self.calibration_rotation = self.plan['rotation']
        self.report.update(tcp_flange_m=self.tcp.tolist(), calibration_tip_high_m=self.plan['above'].tolist(),
                           calibration_tip_low_m=self.plan['grasp'].tolist(),
                           calibration_yaw=self.calibration_yaw,
                           calibration_rpy=self.calibration_rotation.as_euler('xyz').tolist())
        self._save()

    def _reuse_verified_timing(self):
        p = self.previous_timing
        if not p or not p.get('timing_verified') or self.driver_speed != p.get('driver_speed_percent'):
            return False
        if p.get('time_source', 'wall') != ('ros_sim' if getattr(self, 'use_sim_time', False) else 'wall'):
            return False
        if not 0 <= time.time()-p.get('calibrated_at_unix', 0) <= self.timing_max_age:
            return False
        if np.asarray(p.get('tcp_flange_m', [])).shape != (3,) or not np.allclose(p['tcp_flange_m'], self.tcp, atol=1e-6):
            return False
        xy = self.belt.point_of(self.belt.gate_s)
        for field, point in [('calibration_tip_high_m', np.r_[xy, self.belt_z+.122]),
                             ('calibration_tip_low_m', np.r_[xy, self.belt_z+.044])]:
            if np.asarray(p.get(field, [])).shape != (3,) or not np.allclose(p[field], point, atol=.002):
                return False
        rpy = np.asarray(p.get('calibration_rpy', []), float)
        if rpy.shape != (3,) or not np.isfinite(rpy).all():
            return False
        rotation = Rotation.from_euler('xyz', rpy)
        options = yaw_orientations(self.calibration['grasp_rpy'], self.axis, self.calibration_yaw)
        if min((r.inv()*rotation).magnitude() for r in options) > .002:
            return False
        self.calibration_rotation = rotation
        self.timing_verified = True
        self.report.update(timing_verified=True, driver_speed_percent=self.driver_speed,
                           calibrated_at_unix=p['calibrated_at_unix'], descend_seconds=self.descend_seconds,
                           empty_close_seconds=self.close_seconds,
                           **{k: p[k] for k in ('tcp_flange_m', 'calibration_tip_high_m',
                                                'calibration_tip_low_m', 'calibration_rpy')})
        return True

    def _timing_current(self):
        return (self.timing_verified and self.driver_speed == self.report.get('driver_speed_percent')
                and 0 <= time.time()-self.report.get('calibrated_at_unix', 0) <= self.timing_max_age)

    def _calibration_clear(self, now):
        # Require live empty detector output, not an old empty list after camera loss.
        for name in ('_tracks_stamp', '_color_stamp', '_depth_stamp'):
            stamp = getattr(self.node, name, None)
            if stamp is None or not 0 <= now-(stamp.sec+stamp.nanosec*1e-9) <= .2:
                age = None if stamp is None else now-(stamp.sec+stamp.nanosec*1e-9)
                self.report['calibration_not_clear_reason'] = f'{name} age={age} outside [0, 0.2] s'
                self.calibration_empty_since = None
                return False
        u0, v0, u1, v1 = self.roi
        xy = self.belt.point_of(self.belt.gate_s)
        occupied = any((u0 <= tr.u_px <= u1 and v0 <= tr.v_px <= v1)
                       or np.linalg.norm(np.array([tr.centroid_x, tr.centroid_y])-xy) < .12
                       for tr in self.node._tracks)
        if occupied:
            self.report['calibration_not_clear_reason'] = 'Detected object inside calibration area'
            self.calibration_empty_since = None
            return False
        if self.calibration_empty_since is None:
            self.calibration_empty_since = now
        clear = now-self.calibration_empty_since >= .6
        self.report['calibration_not_clear_reason'] = '' if clear else 'Waiting for 0.6 s of clear vision'
        return clear

    def _seed_matches(self):
        joints = self.arm.get_joint_state()
        position, rotation = self._actual()
        return (joints is not None and self.status.motion_status == 0
                and np.max(np.abs(np.asarray(joints[:7])-self.plan['seed_joints'])) < .03
                and np.linalg.norm(position-self.plan['seed_flange']) < .005
                and (rotation.inv()*self.plan['seed_rotation']).magnitude() < .05)

    def _candidate(self, result, tr, now):
        if not np.isfinite([result['stamp'], result['yaw'], result['raw_width_mm'],
                            result['width_mm']]).all():
            raise ValueError('Nonfinite network grasp candidate')
        reference = min(self.history.get(tr.id, [(0., np.zeros(3))]), key=lambda s: abs(s[0]-result['stamp']))
        if abs(reference[0]-result['stamp']) > .06 or not 0 <= now-result['stamp'] <= .3:
            raise ValueError('Network result does not match an accepted image observation')
        center = np.asarray(result['center'], float)
        if (center.shape != (3,) or not np.isfinite(center).all()
                or center[2]-self.offset < tr.belt_z+self.clearance
                or abs(center[2]-reference[1][2]) > .012):
            raise ValueError('Network point not on the observed object surface')
        offset = center[:2]-reference[1][:2]
        if np.linalg.norm(offset) > .035 or result['raw_width_mm']-result['width_mm'] <= 3.:
            raise ValueError('Unstable point offset or insufficient measurable closing margin')
        entries = self.candidates.setdefault(tr.id, [])
        if not entries or result['stamp'] > entries[-1]['stamp']:
            entries.append(dict(stamp=result['stamp'], offset=offset, center=center, result=result))
        entries[:] = [e for e in entries if now-e['stamp'] <= .6]
        if len(entries) < 3:
            return None
        entries = entries[-3:]
        offsets = np.array([e['offset'] for e in entries])
        if np.max(np.linalg.norm(offsets-offsets.mean(axis=0), axis=1)) > .006:
            raise ValueError('Network point changed across frames')
        yaws = np.array([e['result']['yaw'] for e in entries])
        yaw = np.angle(np.mean(np.exp(2j*yaws)))/2
        if max(abs((yaws-yaw+np.pi/2)%np.pi-np.pi/2)) > .15:
            raise ValueError('Network direction changed across frames')
        widths = [e['result']['raw_width_mm'] for e in entries]
        if np.ptp(widths) > 6:
            raise ValueError('Network width changed across frames')
        offset_xy = offsets.mean(axis=0)
        offset_covariance = np.cov(offsets.T)/len(entries)+np.eye(2)*.003**2
        center = center.copy()
        center[:2] += self.belt.d_hat*(self.belt.gate_s-self.belt.s_of(center[:2]))
        return dict(result, center=center, yaw=float(yaw), offset_xy=offset_xy,
                    offset_covariance=offset_covariance)

    def _acquire(self, now):
        u0, v0, u1, v1 = self.roi
        eligible = []
        required = (self.prepare_seconds+self.approach_seconds+self.descend_seconds+self.close_seconds
                    +self.planning_reserve+self.start_margin)
        for tr in self.node._tracks:
            if not (u0+20 < tr.u_px < u1-20 and v0+15 < tr.v_px < v1-15):
                continue
            try:
                remaining = self._remaining(tr, now, np.zeros(2), np.zeros((2, 2)))
            except ValueError:
                continue
            if remaining < required or self.rejected_until.get(tr.id, 0.) > time.monotonic():
                continue
            eligible.append((remaining, tr))
        owner = next((tr for _, tr in eligible if tr.id == self.acquisition_id), None)
        if owner is not None and time.monotonic()-self.acquisition_started > 3.:
            self.rejected_until[owner.id] = time.monotonic()+3.
            eligible = [(remaining, tr) for remaining, tr in eligible if tr.id != owner.id]
            owner = None
        if owner is None:
            self.acquisition_id = None
            if not eligible:
                return
            owner = min(eligible, key=lambda item: item[0])[1]
            self.acquisition_id = owner.id
            self.acquisition_started = time.monotonic()
        tr = self.track = owner
        self.node._fsm.target_id, self.node._fsm.state = tr.id, 'TRACK'
        self.node._maybe_request_inference(self.node.get_clock().now())
        result = self.node._inference.latest(max_age=.5) if self.node._inference else None
        if result is None or result.get('status') != 0 or result.get('target_id') != tr.id:
            return
        try:
            candidate = self._candidate(result, tr, now)
            if candidate is None:
                return
            if self._remaining(tr, now, candidate['offset_xy'], candidate['offset_covariance']) < required:
                raise ValueError('Insufficient time for planning and complete motion')
        except ValueError as error:
            self.node.get_logger().warning(str(error), throttle_duration_sec=3.)
            return
        # Offset belongs to the selected target, never to a rejected candidate.
        self.offset_xy = candidate['offset_xy'].copy()
        self.offset_covariance = candidate['offset_covariance'].copy()
        center = candidate['center']
        trial_track = SimpleNamespace(centroid_x=center[0], centroid_y=center[1], belt_z=tr.belt_z, id=tr.id)
        self.report.update(track_id=tr.id, network_score=result['score'],
                           network_offset_xy_m=self.offset_xy.tolist(),
                           network_offset_covariance=self.offset_covariance.tolist(),
                           network_stamp=result['stamp'], preparation_budget_s=required)
        self.node._publish_candidate(result)
        self.planning_future = self.planner.submit(self._make_plan, candidate, trial_track, .06)
        self._set('CONVEYOR_PLANNING')

    def _planning_complete(self, now):
        try:
            plan = self.planning_future.result()
            self.plan = plan
            if abs((plan['above'][2]-plan['grasp'][2])-self.stroke) > .002:
                raise ValueError('Candidate stroke differs from measured descend stroke')
            self.contact_seconds = contact_delay(self.close_seconds, self.open_width,
                                                   self.calibration_goal, plan['raw_width'])
            self.horizon = self.descend_seconds+self.contact_seconds
            tr = self._target(now)
            if self._remaining(tr, now) < self.prepare_seconds+self.approach_seconds+self.horizon:
                raise ValueError('Target arrived too late after IK preflight')
            self.report.update(descend_contact_horizon_s=self.horizon, tip_grasp_m=plan['grasp'].tolist())
            # Deliberately no motion: wait for the computed upstream start region.
            self._set('CONVEYOR_WAIT_START')
        except ValueError as error:
            self._skip(str(error))

    def _wait_start(self, now):
        try:
            tr = self._target(now)
            remaining = self._remaining(tr, now)
            window = stage_window(max(0., remaining), self.prepare_seconds+self.approach_seconds, self.descend_seconds,
                                  self.contact_seconds, self.start_margin)
            if remaining < 0 or window == 'LATE':
                self._skip('Target passed the feasible startup region'); return
            self.report['startup_distance_m'] = float(
                (np.asarray(tr.velocity)[:2]@self.belt.d_hat)*
                (self.prepare_seconds+self.approach_seconds+self.horizon+self.start_margin))
            self.report['remaining_to_contact_s'] = remaining
            # Forecast the whole time until interception, including any gate wait.
            estimate, bounds = self._prediction(tr, now, max(remaining, 0.))
            if not self._capture_ok(estimate, bounds):
                self.stable_count, self.stable_stamp = 0, None
                return
            self._stable(tr)
            if window != 'START' or self.stable_count < self.stable_frames:
                return
            if not self._execution_enabled():
                self.report['preview_would_start'] = True; self._save(); return
            if not self._timing_current():
                self._set('CONVEYOR_NEED_CALIBRATION'); return
            if (self.calibration_rotation is None
                    or (self.calibration_rotation.inv()*self.plan['rotation']).magnitude() > .15):
                self._skip('Candidate orientation requires a new descend timing measurement'); return
            calibration_xy = self.belt.point_of(self.belt.gate_s)
            if np.linalg.norm(self.plan['grasp'][:2]-calibration_xy) > .05:
                self._skip('Candidate position outside timing calibration neighborhood'); return
            if not self._seed_matches():
                self._skip('Arm moved after path preflight; new planning required'); return
            if not self.enable.service_is_ready():
                raise ValueError('Enable service unavailable')
            request = SetBool.Request(); request.data = True
            self.enable_future = self.enable.call_async(request)
            self.report['calibration_expected_speed'] = self.driver_speed
            self.startup_started = self._motion_time()
            self.motion_issued = True
            self._set('CONVEYOR_START_ENABLE')
        except ValueError as error:
            self._skip(str(error))

    def _wait_gate(self, now):
        try:
            tr = self._target(now)
            remaining = self._remaining(tr, now)
            window = stage_window(max(0., remaining), 0., self.descend_seconds,
                                  self.contact_seconds, self.trigger_margin)
            if remaining < 0 or window == 'LATE':
                self._skip('Too late for calibrated descent'); return
            estimate, bounds = self._prediction(tr, now, self.horizon)
            if not self._capture_ok(estimate, bounds):
                self.stable_count, self.stable_stamp = 0, None
                return
            if not self._stable(tr) or window != 'START':
                return
            if not self._execution_enabled():
                return
            self.report['trigger_prediction_xy_m'] = estimate['position'].tolist()
            self.report['trigger_bounds_m'] = {k: float(v) for k, v in bounds.items()}
            self._motion(self.plan['grasp'], 'CONVEYOR_DESCEND')
        except ValueError as error:
            self._skip(str(error))

    def tick(self):
        if self.state in ('CONVEYOR_DONE', 'CONVEYOR_ABORT', 'CONVEYOR_CALIB_DONE'):
            return
        try:
            enabled = self._execution_enabled()
            if not enabled and self.motion_issued:
                raise ValueError('Execution disabled during active motion')
            if not self._fresh():
                if self.motion_issued:
                    raise ValueError('Feedback stale')
                return
            if self.status.err_status or self.status.arm_status in (1, 5, 7, 9, 10):
                raise ValueError('Driver fault '+str(self.status.arm_status))
            now = self.node.get_clock().now().nanoseconds*1e-9
            self._observe(now)
            elapsed = self._motion_time()-self.started
            if (enabled and self.state.startswith('CONVEYOR_CALIB_')
                    and self.state != 'CONVEYOR_CALIB_READY' and not self._calibration_clear(now)):
                raise ValueError('Timing calibration area occupied or vision stale')
            if enabled and self.state not in ('CONVEYOR_CALIB_READY', 'CONVEYOR_CALIB_ENABLE'):
                if not self._driver_profile():
                    if self.motion_issued: raise ValueError('Driver motion profile unavailable')
                    return
                if self.timing_verified and self.driver_speed != self.report.get('driver_speed_percent'):
                    raise ValueError('Driver speed changed after timing measurement')
                if (self.state in self.ACTIVE_STATES and self.report.get('calibration_expected_speed')
                        and self.driver_speed != self.report['calibration_expected_speed']):
                    raise ValueError('Driver speed changed during active motion')
            if self.state == 'CONVEYOR_CALIB_READY':
                if not self.calibration_only:
                    self._set('CONVEYOR_NEED_CALIBRATION'); return
                if not enabled:
                    self.report['calibration_preview'] = True
                    return
                if not self._driver_profile() or not self._calibration_clear(now):
                    return
                if self._reuse_verified_timing():
                    self._set('CONVEYOR_CALIB_DONE'); return
                if self.calibration_future is not None:
                    if not self.calibration_future.done():
                        return
                    self.calibration_future.result()
                    self.calibration_future = None
                    # Recheck fresh feedback and clear vision on the next tick.
                    return
                if self.plan is None:
                    # Keep image, clock and feedback callbacks running during IK.
                    self.calibration_future = self.planner.submit(self._calibration_plan)
                    return
                now = self.node.get_clock().now().nanoseconds*1e-9
                if not self._calibration_clear(now):
                    return
                if not self._seed_matches():
                    self.plan = None
                    self.report['calibration_not_clear_reason'] = 'Arm moved after calibration preflight'
                    return
                self.report['calibration_expected_speed'] = self.driver_speed
                if not self.enable.service_is_ready(): raise ValueError('Enable service unavailable')
                request = SetBool.Request(); request.data = True
                self.enable_future = self.enable.call_async(request)
                self.motion_issued = True
                self._set('CONVEYOR_CALIB_ENABLE')
            elif self.state == 'CONVEYOR_CALIB_ENABLE':
                if self.enable_future.done():
                    if not self.enable_future.result().success: raise ValueError('Arm enable failed')
                    self._grip(self.open_width, 'CONVEYOR_CALIB_OPEN')
                elif elapsed > 8: raise ValueError('Enable timed out')
            elif self.state in ('CONVEYOR_CALIB_OPEN', 'CONVEYOR_CALIB_REOPEN'):
                width = self.arm.get_gripper_width()
                if elapsed > 1 and width is not None and width >= self.open_width-.003:
                    if self.state == 'CONVEYOR_CALIB_OPEN':
                        self._motion(self.plan['above'], 'CONVEYOR_CALIB_APPROACH', self.plan['joints'][0])
                    else: self._motion(self.plan['above'], 'CONVEYOR_CALIB_RAISE')
                elif elapsed > 4: raise ValueError('Gripper open feedback failed')
            elif self.state == 'CONVEYOR_CALIB_APPROACH' and self._arrived():
                if self.timing_verified:
                    self.plan = None; self._set('CONVEYOR_CALIB_DONE')
                else: self._motion(self.plan['grasp'], 'CONVEYOR_CALIB_DESCEND')
            elif self.state == 'CONVEYOR_CALIB_DESCEND' and self._arrived():
                self.descend_seconds = self._motion_time()-self.started
                if not .4 < self.descend_seconds < 8: raise ValueError('Unexpected descend time')
                self.report['descend_seconds'] = self.descend_seconds
                self._grip(self.calibration_goal, 'CONVEYOR_CALIB_CLOSE')
            elif self.state == 'CONVEYOR_CALIB_CLOSE':
                width = self.arm.get_gripper_width()
                if elapsed > .2 and width is not None and abs(width-self.calibration_goal) < .002:
                    self.close_seconds = elapsed
                    self.timing_verified = True
                    self.report.update(empty_close_seconds=elapsed, timing_verified=True,
                                       driver_speed_percent=self.driver_speed, calibrated_at_unix=time.time())
                    self._grip(self.open_width, 'CONVEYOR_CALIB_REOPEN')
                elif elapsed > 4: raise ValueError('Empty close blocked or timed out')
            elif self.state == 'CONVEYOR_CALIB_RAISE' and self._arrived():
                self.plan = None
                self._set('CONVEYOR_CALIB_DONE' if self.calibration_only else 'CONVEYOR_WAIT_TARGET')
            elif self.state == 'CONVEYOR_WAIT_TARGET':
                if enabled and not self._timing_current() and not self._reuse_verified_timing():
                    self.report['calibration_required_reason'] = 'No current verified timing for this geometry and driver speed'
                    self._set('CONVEYOR_NEED_CALIBRATION')
                elif self.node._inference is None: raise ValueError('No GR-ConvNet model; no fallback')
                else: self._acquire(now)
            elif self.state == 'CONVEYOR_NEED_CALIBRATION':
                if not enabled:
                    self._set('CONVEYOR_WAIT_TARGET')
            elif self.state == 'CONVEYOR_PLANNING' and self.planning_future.done():
                self._planning_complete(now)
            elif self.state == 'CONVEYOR_WAIT_START': self._wait_start(now)
            elif self.state in ('CONVEYOR_START_ENABLE', 'CONVEYOR_START_OPEN', 'CONVEYOR_APPROACH'):
                tr = self._target(now)
                remaining = self._remaining(tr, now)
                estimate, bounds = self._prediction(tr, now, max(remaining, 0.))
                if remaining < self.horizon or not self._capture_ok(estimate, bounds):
                    raise ValueError('Target became infeasible during startup')
                startup_elapsed = self._motion_time()-self.startup_started
                if startup_elapsed > self.prepare_seconds+self.approach_seconds+self.start_margin:
                    raise ValueError('Approach exceeded interception time budget')
                if self.state == 'CONVEYOR_START_ENABLE':
                    if self.enable_future.done():
                        if not self.enable_future.result().success:
                            raise ValueError('Arm enable failed')
                        self._grip(self.open_width, 'CONVEYOR_START_OPEN')
                elif self.state == 'CONVEYOR_START_OPEN':
                    width = self.arm.get_gripper_width()
                    if elapsed > 1 and width is not None and width >= self.open_width-.003:
                        if not self._seed_matches():
                            raise ValueError('Arm moved while enabling; path preflight invalid')
                        if remaining < self.approach_seconds+self.horizon:
                            raise ValueError('Insufficient approach time after opening')
                        self._motion(self.plan['above'], 'CONVEYOR_APPROACH', self.plan['joints'][0])
                elif self._arrived():
                    self.report['actual_approach_seconds'] = elapsed
                    self.report['actual_prepare_approach_seconds'] = startup_elapsed
                    self._set('CONVEYOR_WAIT_GATE')
            elif self.state == 'CONVEYOR_WAIT_GATE': self._wait_gate(now)
            elif self.state == 'CONVEYOR_DESCEND':
                tr = self._target(now, allow_lost=True)
                estimate, bounds = self._prediction(tr, now, max(self.horizon-elapsed, 0), allow_lost=True)
                if not self._capture_ok(estimate, bounds):
                    raise ValueError('Contact confidence bounds left the grasp region')
                if elapsed > self.descend_seconds+.5: raise ValueError('Descend exceeded calibrated timing')
                if self._arrived():
                    self.report['actual_descend_seconds'] = self._motion_time()-self.started
                    self._grip(self.plan['close_width'], 'CONVEYOR_CLOSE')
            elif self.state == 'CONVEYOR_CLOSE' and elapsed > max(1.5, self.close_seconds+.3):
                width = self.arm.get_gripper_width(); self.report['actual_closed_width_m'] = width
                if width is None or width <= self.plan['close_width']+.003 or width >= self.open_width-.01:
                    self.failure = 'Empty or unverified grasp'; self._grip(self.open_width, 'CONVEYOR_RECOVER_OPEN')
                else: self._motion(self.plan['lift'], 'CONVEYOR_LIFT')
            elif self.state == 'CONVEYOR_LIFT' and self._arrived(): self._set('CONVEYOR_HOLD')
            elif self.state == 'CONVEYOR_HOLD' and elapsed > 2: self._motion(self.plan['grasp'], 'CONVEYOR_RETURN')
            elif self.state == 'CONVEYOR_RETURN' and self._arrived(): self._grip(self.open_width, 'CONVEYOR_RELEASE')
            elif self.state in ('CONVEYOR_RELEASE', 'CONVEYOR_RECOVER_OPEN') and elapsed > 1.5:
                self._motion(self.plan['above'], 'CONVEYOR_RECOVER_RAISE' if self.failure else 'CONVEYOR_RETRACT')
            elif self.state in ('CONVEYOR_RETRACT', 'CONVEYOR_RECOVER_RAISE') and self._arrived():
                self.report['failure'] = self.failure
                self._set('CONVEYOR_ABORT' if self.failure else 'CONVEYOR_DONE')
        except Exception as error:
            self.report['failure'] = str(error)
            self._abort(str(error))
