"""One conveyor-follow pick with early planning and measured speed admission.

Preview is the default. AGX move_js streams bounded joint position references;
it must have exactly one subscriber and no competing stream publisher.
No camera/belt ground-truth speed is used. A completed pick holds the object.
"""
import json
import time
from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Pose, PoseArray
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool
from agx_arm_msgs.msg import GripperStatus

from .static_once import StaticOnce
from .kinematics import yaw_orientations
from .plan_validity import seed_matches
from .conveyor_prediction import motion_at_contact, forecast_motion
from .follow_math import (vector, region_box, in_region, interception_window,
    smooth_segment, brake_displacement, TipVelocity, matched, resolved_rate,
    preflight_follow, flange_in_workspace, arm_fault)


class ConveyorFollow(StaticOnce):
    TERMINAL = {'FOLLOW_DONE', 'FOLLOW_ABORT'}
    SERVO_STATES = {'FOLLOW_SYNC', 'FOLLOW_DESCEND', 'FOLLOW_CLOSE',
                    'FOLLOW_LIFT', 'FOLLOW_BRAKE'}

    def __init__(self, node):
        super().__init__(node)
        declare = lambda name, default: node.declare_parameter(
            name, getattr(node, '_mode_params', {}).get(name, default)).value
        ws = node._p['workspace']
        self.region = vector(declare('follow_region',
            [ws['x_min'], ws['x_max'], ws['y_min'], ws['y_max']]), 4, 'follow region')
        self.margin = float(declare('follow_region_margin', .02))
        region_box(self.region, self.margin)
        self.prepare = float(declare('follow_prepare_seconds', 1.5))
        self.approach = float(declare('follow_approach_seconds', 2.))
        self.reserve = float(declare('follow_planning_reserve', 2.5))
        self.sync_timeout = float(declare('follow_sync_timeout', 1.5))
        self.descend = float(declare('follow_descend_seconds', 2.5))
        self.close_timeout = float(declare('follow_close_timeout', 1.2))
        self.lift_seconds = float(declare('follow_lift_seconds', 2.5))
        self.brake_seconds = float(declare('follow_brake_seconds', .6))
        self.high_offset = float(declare('follow_approach_height', .08))
        self.lift_offset = float(declare('follow_lift_height', .10))
        self.latency = float(declare('follow_command_latency', .03))
        self.max_occlusion = float(declare('follow_close_occlusion', .35))
        self.speed_min = float(declare('follow_speed_min', .008))
        self.speed_max = float(declare('follow_speed_max', .04))
        self.velocity_tol = float(declare('follow_velocity_tolerance', .008))
        self.position_tol = float(declare('follow_position_tolerance', .016))
        self.cross_tol = float(declare('follow_cross_tolerance', .012))
        self.gain = float(declare('follow_position_gain', 4.))
        self.joint_speed = float(declare('follow_joint_speed', .6))
        self.joint_accel = float(declare('follow_joint_accel', 2.))
        self.linear_speed = float(declare('follow_linear_speed', .10))
        self.angular_speed = float(declare('follow_angular_speed', .3))
        self.sigma = float(declare('follow_confidence_sigma', 3.))
        self.plan_file = declare('follow_plan_file', '/tmp/nero7_follow_plan.json')
        positives = [self.prepare, self.approach, self.reserve, self.sync_timeout,
            self.descend, self.close_timeout, self.lift_seconds, self.brake_seconds,
            self.high_offset, self.lift_offset, self.speed_min, self.speed_max,
            self.velocity_tol, self.position_tol, self.cross_tol, self.gain,
            self.joint_speed, self.joint_accel, self.linear_speed, self.angular_speed, self.sigma]
        if (not np.isfinite(positives+[self.latency, self.max_occlusion]).all()
                or min(positives) <= 0 or not 0 <= self.latency <= .1
                or not 0 < self.max_occlusion <= .4
                or not self.speed_min < self.speed_max <= .04
                or node._p['fsm_rate'] < 50 or node._p['backend'] != 'ros_topic'):
            raise ValueError('Invalid follow limits; requires ros_topic and fsm_rate >= 50')
        # Quintic vertical motion peaks at 1.875 * stroke/duration.
        if np.hypot(max(1.875*(self.high_offset+self.offset)/self.descend,
                        1.875*self.lift_offset/self.lift_seconds),
                    np.hypot(self.speed_max, .012)) > self.linear_speed:
            raise ValueError('Following/vertical speed budget exceeds linear limit')
        self.belt = node._belt
        self.state = 'FOLLOW_WAIT_TARGET'
        self.history, self.candidates = {}, {}
        self.offset_xy = np.zeros(2)
        self.offset_covariance = np.eye(2)*.003**2
        self.velocity_feedback = TipVelocity()
        self.last_pose_stamp, self.last_joint_stamp, self.last_grip_stamp = None, None, None
        self.joint_sample, self.joint_received, self.grip_fault = None, 0., False
        self.grip_width, self.grip_time, self.grip_counter = None, 0., 0
        self.grip_verified_since, self.grip_verified_count = None, 0
        self.sync_since, self.sync_pose_stamp, self.sync_frames = None, None, 0
        self.last_tick = None
        self.q_command, self.q_velocity = None, np.zeros(7)
        self.frozen, self.grip_confirmed = None, False
        self.last_preview, self.last_infer = -float('inf'), -float('inf')
        self.last_skip, self.retry_at = None, 0.
        self.rejected_until, self.acquisition_at = {}, 0.
        self.grip_blocked_frames = 0
        self.report = dict(mode='conveyor_follow', estimator='Kalman XY CV',
            region_base_xy=self.region.tolist(), execution_default=False,
            events=[], diagnostics={}, trajectory=[])
        self.pub_preview = node.create_publisher(String, '/grasp/follow/diagnostics', 1)
        self.pub_path = node.create_publisher(PoseArray, '/grasp/follow/trajectory', 1)
        node.create_subscription(JointState, node._p['topic_prefix']+'feedback/joint_states',
                                 self._joint_feedback, 5)
        node.create_subscription(GripperStatus, node._p['topic_prefix']+'feedback/gripper_status',
                                 self._gripper_feedback, 5)

    @property
    def duration(self):
        return self.sync_timeout+self.descend+self.close_timeout+self.lift_seconds+self.brake_seconds

    def _set(self, state):
        super()._set(state.replace('STATIC_', 'FOLLOW_'))
        self.sync_since, self.sync_pose_stamp = None, None
        self.report['events'].append(dict(state=self.state, at_ros=self.node.get_clock().now().nanoseconds*1e-9))
        self.report['events'] = self.report['events'][-60:]

    def _abort(self, reason):
        self.failure = str(reason)
        self.node.get_logger().error('FOLLOW_ABORT: '+self.failure)
        if self.motion_issued:
            self.arm.estop()
        self._set('FOLLOW_ABORT')
        self.report['failure'] = self.failure

    def _pose_cb(self, msg):
        if not hasattr(self, 'velocity_feedback'):
            super()._pose_cb(msg)
            return
        stamp = msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        now = self.node.get_clock().now().nanoseconds*1e-9
        if (msg.header.frame_id not in ('', 'base_link') or not 0 <= now-stamp <= .15
                or (self.last_pose_stamp is not None and stamp <= self.last_pose_stamp)):
            return
        p, q = msg.pose.position, msg.pose.orientation
        try:
            point = vector([p.x, p.y, p.z], 3, 'flange feedback')
            quaternion = vector([q.x, q.y, q.z, q.w], 4, 'flange orientation')
            if abs(np.linalg.norm(quaternion)-1.) > 1e-3:
                return
            rotation = Rotation.from_quat(quaternion)
            if self.velocity_feedback.add(stamp, point+rotation.apply(self.tcp)):
                # Keep an independent accepted sample; a reused/rejected ROS
                # message object must not mutate the controller's current pose.
                accepted = Pose()
                accepted.position.x, accepted.position.y, accepted.position.z = map(float, point)
                (accepted.orientation.x, accepted.orientation.y,
                 accepted.orientation.z, accepted.orientation.w) = map(float, quaternion)
                self.pose, self.pose_time, self.last_pose_stamp = accepted, time.monotonic(), stamp
        except ValueError:
            return

    def _joint_feedback(self, msg):
        stamp = msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        now = self.node.get_clock().now().nanoseconds*1e-9
        if (not 0 <= now-stamp <= .15 or len(msg.name) != len(msg.position)
                or len(set(msg.name)) != len(msg.name)
                or (self.last_joint_stamp is not None and stamp <= self.last_joint_stamp)):
            return
        mapping = dict(zip(msg.name, msg.position))
        names = [f'joint{i}' for i in range(1, 8)]
        if not all(name in mapping for name in names):
            return
        try:
            sample = vector([mapping[name] for name in names], 7, 'joint feedback')
        except ValueError:
            return
        self.joint_sample, self.last_joint_stamp = sample, stamp
        self.joint_received = time.monotonic()

    def _joints(self):
        return vector(self.joint_sample, 7, 'accepted joint feedback').copy()

    def _gripper_feedback(self, msg):
        stamp = msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        now = self.node.get_clock().now().nanoseconds*1e-9
        fault = any(getattr(msg, flag, False) for flag in (
            'voltage_too_low', 'motor_overheating', 'driver_overcurrent',
            'driver_overheating', 'sensor_status', 'driver_error_status'))
        if (not 0 <= now-stamp <= .15
                or (self.last_grip_stamp is not None and stamp <= self.last_grip_stamp)):
            return
        self.grip_fault = bool(fault)
        width = float(msg.width)
        if np.isfinite(width) and 0 <= width <= self.open_width+.005:
            self.grip_width, self.grip_time, self.last_grip_stamp = width, time.monotonic(), stamp
            self.grip_counter += 1

    def _gripper_fresh(self):
        now = self.node.get_clock().now().nanoseconds*1e-9
        return (self.grip_width is not None and self.last_grip_stamp is not None
                and time.monotonic()-self.grip_time < .2
                and 0 <= now-self.last_grip_stamp <= .15)

    def _fresh(self):
        if (not super()._fresh() or self.last_pose_stamp is None
                or self.last_joint_stamp is None):
            return False
        now = self.node.get_clock().now().nanoseconds*1e-9
        return (0 <= now-self.last_pose_stamp <= .15
                and 0 <= now-self.last_joint_stamp <= .15
                and time.monotonic()-self.joint_received < .2 and self._gripper_fresh())

    def _observe(self, now):
        for tr in self.node._tracks:
            if tr.status not in (0, 1) or not 0 <= now-tr.stamp <= .15:
                continue
            samples = self.history.setdefault(tr.id, deque(maxlen=150))
            if not samples or tr.stamp > samples[-1][0]:
                samples.append((tr.stamp, tr.observed_position.copy()))
            while samples and now-samples[0][0] > 3.:
                samples.popleft()
        for tid in list(self.history):
            if not self.history[tid] or now-self.history[tid][-1][0] > 1.:
                self.history.pop(tid)
                self.candidates.pop(tid, None)

    def _request(self, now):
        if self.track is not None and now-self.last_infer >= .1:
            self.node._fsm.target_id, self.node._fsm.state = self.track.id, 'TRACK'
            self.node._maybe_request_inference(self.node.get_clock().now())
            self.last_infer = now

    def _candidate(self, result, tr, now):
        if result is None or result.get('status') != 0 or result.get('target_id') != tr.id:
            return None
        fields = [result['stamp'], result['yaw'], result['raw_width_mm'], result['width_mm']]
        if not np.isfinite(fields).all() or not 0 <= now-result['stamp'] <= .3:
            return None
        history = self.history.get(tr.id, [])
        if not history:
            return None
        reference = min(history, key=lambda sample: abs(sample[0]-result['stamp']))
        if abs(reference[0]-result['stamp']) > .06:
            return None
        center = vector(result['center'], 3, 'network grasp point')
        offset = center[:2]-reference[1][:2]
        if (np.linalg.norm(offset) > .035 or abs(center[2]-reference[1][2]) > .012
                or center[2]-self.offset < tr.belt_z+self.clearance
                or not .012 < result['raw_width_mm']/1000 < self.open_width-.002
                or not 0 < result['width_mm'] < result['raw_width_mm']-3.):
            raise ValueError('Network point/width is inconsistent with the observed object')
        entries = self.candidates.setdefault(tr.id, deque(maxlen=3))
        if not entries or result['stamp'] > entries[-1]['stamp']:
            entries.append(dict(stamp=result['stamp'], offset=offset, center=center, result=result))
        if len(entries) < 3 or now-entries[0]['stamp'] > .6:
            return None
        offsets = np.array([entry['offset'] for entry in entries])
        yaws = np.array([entry['result']['yaw'] for entry in entries])
        yaw = float(np.angle(np.mean(np.exp(2j*yaws)))/2)
        widths = np.array([entry['result']['raw_width_mm'] for entry in entries])
        if (np.max(np.linalg.norm(offsets-offsets.mean(axis=0), axis=1)) > .006
                or np.max(abs((yaws-yaw+np.pi/2)%np.pi-np.pi/2)) > .15
                or np.ptp(widths) > 6.
                or np.ptp([entry['center'][2] for entry in entries]) > .008):
            raise ValueError('Grasp offset/orientation/width is unstable')
        return dict(result, yaw=yaw, offset_xy=offsets.mean(axis=0),
            offset_covariance=np.cov(offsets.T)/3+np.eye(2)*.003**2)

    def _estimate(self, tr, now, horizon=0., allow_lost=False):
        estimate = motion_at_contact(tr, now, horizon, self.offset_xy,
                                     self.offset_covariance, 0., allow_lost)
        v = estimate['velocity']
        along = float(v@self.belt.d_hat)
        if not self.speed_min <= along <= self.speed_max or np.linalg.norm(v-along*self.belt.d_hat) > .012:
            raise ValueError('Qualified motion is outside the follow trial limits')
        if not 0 <= now-tr.stamp <= .15:
            raise ValueError('Follow observation is stale')
        return estimate

    def _target(self, now, allow_lost=False):
        tr = next((t for t in self.node._tracks if self.track is not None and t.id == self.track.id), None)
        if tr is None:
            raise ValueError('Bound following target disappeared')
        estimate = self._estimate(tr, now, self.latency, allow_lost)
        if abs(tr.centroid_z-self.plan['surface_z']) > .012:
            raise ValueError('Object height changed after preflight')
        return tr, estimate

    def _skip(self, reason, now):
        self.last_skip, self.retry_at = str(reason), now+.5
        self.report['last_skip'] = self.last_skip
        self.node.get_logger().warning('Follow target skipped: '+self.last_skip, throttle_duration_sec=2.)
        if self.track is not None:
            self.rejected_until[self.track.id] = now+1.
        self.plan = None
        self.planning_future = None
        self.track = None
        self._set('FOLLOW_WAIT_TARGET')

    def _acquire(self, now):
        visible = [tr for tr in self.node._tracks if tr.status in (0, 1)
                   and 0 <= now-tr.stamp <= .15 and self.rejected_until.get(tr.id, 0.) <= now]
        # Retain the visible ID while its speed converges. No grip-region filter
        # here: request inference from the first usable image upstream.
        owner = next((tr for tr in visible if self.track is not None and tr.id == self.track.id), None)
        if owner is not None and not owner.speed_valid and now-self.acquisition_at > 3.:
            self.rejected_until[owner.id] = now+1.
            visible = [tr for tr in visible if tr.id != owner.id]
            owner = None
        if owner is None:
            self.plan = None
            if not visible:
                self.track = None
                return
            owner = min(visible, key=lambda tr: self.belt.s_of([tr.centroid_x, tr.centroid_y]))
            self.acquisition_at = now
            self.offset_xy, self.offset_covariance = np.zeros(2), np.eye(2)*.003**2
        self.track = owner
        self._request(now)
        self.report['diagnostics'].update(track_id=owner.id, speed_valid=bool(owner.speed_valid),
            planning_from_first_detection=True, observed_age_s=now-owner.stamp)
        if owner.status != 1 or not owner.speed_valid or now < self.retry_at:
            return
        if self.node._inference is None:
            raise ValueError('GR-ConvNet unavailable; no grasp fallback')
        result = self.node._inference.latest(max_age=.5)
        try:
            candidate = self._candidate(result, owner, now)
            if candidate is None:
                return
            self.offset_xy = candidate['offset_xy'].copy()
            self.offset_covariance = candidate['offset_covariance'].copy()
            estimate = self._estimate(owner, now)
            window = interception_window(estimate['position'], estimate['velocity'], self.region,
                self.prepare+self.approach+self.reserve, self.duration, self.margin)
            seed = vector(self._joints(), 7, 'plan seed')
            position, rotation = self._actual()
            snapshot = dict(id=owner.id, belt_z=float(owner.belt_z),
                estimate=estimate, window=window, planned_at=now, seed=seed.copy(),
                seed_flange=position.copy(), seed_rotation=rotation,
                candidate=candidate, surface_z=float(candidate['center'][2]))
            self.planning_future = self.static_planner.submit(self._build_plan, snapshot)
            self._set('FOLLOW_PLANNING')
        except ValueError as error:
            self._skip(error, now)

    def _build_plan(self, data):
        seed, candidate, estimate = data['seed'], data['candidate'], data['estimate']
        T = self.kin.fk(seed)
        if (np.linalg.norm(T[:3, 3]-data['seed_flange']) > .005
                or (Rotation.from_matrix(T[:3, :3]).inv()*data['seed_rotation']).magnitude() > .05):
            raise ValueError('URDF does not agree with actual flange feedback')
        low = data['surface_z']-self.offset
        if low < data['belt_z']+self.clearance:
            raise ValueError('Follow grasp violates belt clearance')
        high = data['surface_z']+self.high_offset
        start = data['window']['start']
        xy = estimate['position']+estimate['velocity']*start
        count = int(np.ceil(self.duration/.25))
        times = np.linspace(0., self.duration, count+1)
        dt = self.duration/count
        tips = []
        descend_end = self.sync_timeout+self.descend
        close_end = descend_end+self.close_timeout
        lift_end = close_end+self.lift_seconds
        for t in times:
            point_xy = xy+estimate['velocity']*min(t, lift_end)
            if t <= self.sync_timeout:
                z = high
            elif t <= descend_end:
                z, _ = smooth_segment(high, low, t-self.sync_timeout, self.descend)
            elif t <= close_end:
                z = low
            elif t <= lift_end:
                z, _ = smooth_segment(low, low+self.lift_offset, t-close_end, self.lift_seconds)
            else:
                delta, _ = brake_displacement(estimate['velocity'], t-lift_end, self.brake_seconds)
                point_xy += delta
                z = low+self.lift_offset
            if not in_region(point_xy, self.region, self.margin):
                raise ValueError('Nominal following trajectory leaves grasp region')
            tips.append(np.r_[point_xy, z])
        tips = np.asarray(tips)
        options = []
        for rotation in yaw_orientations(self.calibration['grasp_rpy'], self.axis, candidate['yaw']):
            try:
                solutions = preflight_follow(self.kin, seed, self.tcp, rotation, tips,
                    self.node._p['workspace'], data['belt_z'], self.joint_speed, dt)
                if np.max(np.abs(solutions[0]-seed)) > 2.3:
                    continue
                for alpha in np.linspace(0., 1., 100):
                    T = self.kin.fk(seed*(1-alpha)+solutions[0]*alpha)
                    tip = T[:3, 3]+T[:3, :3]@self.tcp
                    if tip[2] < data['belt_z']+.05:
                        raise ValueError('Joint approach crosses the belt')
                options.append((np.linalg.norm(solutions[0]-seed), rotation, solutions))
            except ValueError:
                continue
        if not options:
            raise ValueError('No verified IK/clear trajectory for the complete following pick')
        _, rotation, solutions = min(options, key=lambda option: option[0])
        return dict(track_id=data['id'], seed_joints=seed, seed_flange=data['seed_flange'],
            seed_rotation=data['seed_rotation'], rotation=rotation, joints=solutions,
            above=tips[0], grasp=np.r_[tips[0, :2], low], high_z=high, low_z=low,
            surface_z=data['surface_z'], belt_z=data['belt_z'], raw_width=candidate['raw_width_mm']/1000,
            close_width=candidate['width_mm']/1000, offset_xy=candidate['offset_xy'],
            offset_covariance=candidate['offset_covariance'], yaw=candidate['yaw'],
            planned_at=data['planned_at'], sync_at=data['planned_at']+start,
            reference_xy=estimate['position'], reference_velocity=estimate['velocity'],
            image_stamp=candidate['stamp'], path=tips, path_times=times)

    def _ready_plan(self, now, before_motion=True):
        if self.plan is None or self.plan['track_id'] != self.track.id:
            raise ValueError('Follow plan belongs to another target')
        tr, estimate = self._target(now)
        predicted = self.plan['reference_xy']+self.plan['reference_velocity']*(now+self.latency-self.plan['planned_at'])
        if (np.linalg.norm(predicted-estimate['position']) > .02
                or np.linalg.norm(estimate['velocity']-self.plan['reference_velocity']) > .012):
            raise ValueError('Object trajectory changed since preflight')
        remaining = self.plan['sync_at']-now
        requested = max(0., remaining-self.latency)
        window = interception_window(estimate['position'], estimate['velocity'], self.region,
                                     requested, self.duration, self.margin)
        if window['start'] > requested+1e-6:
            raise ValueError('Object will enter the region after the planned synchronization')
        if before_motion and self.status.motion_status != 0:
            raise ValueError('Arm is moving before the verified following approach')
        result = self.node._inference.latest(max_age=.5) if self.node._inference else None
        candidate = self._candidate(result, tr, now)
        if candidate is not None and candidate['stamp'] > self.plan['image_stamp']:
            if (np.linalg.norm(candidate['offset_xy']-self.plan['offset_xy']) > .008
                    or abs((candidate['yaw']-self.plan['yaw']+np.pi/2)%np.pi-np.pi/2) > .15
                    or abs(candidate['raw_width_mm']/1000-self.plan['raw_width']) > .006):
                raise ValueError('Object-relative grasp changed after preflight')
        if before_motion:
            position, rotation = self._actual()
            if not seed_matches(self.plan, self._joints(), position, rotation):
                raise ValueError('Arm moved after follow preflight')
        return tr, estimate

    def _save_plan(self):
        p = self.plan
        output = dict(mode='conveyor_follow', track_id=p['track_id'], generated_at_unix=time.time(),
            sync_at_ros=p['sync_at'], image_stamp=p['image_stamp'], region=self.region.tolist(),
            measured_object_velocity_m_s=p['reference_velocity'].tolist(),
            fingertip_trajectory_m=p['path'].tolist(), trajectory_times_s=p['path_times'].tolist(),
            grasp_rpy=p['rotation'].as_euler('xyz').tolist(), tcp_flange_m=self.tcp.tolist(),
            opening_m=self.open_width, closing_m=p['close_width'])
        with open(self.plan_file, 'w') as stream:
            json.dump(output, stream, indent=2)

    def _preview(self, now):
        if now-self.last_preview < .2:
            return
        self.last_preview = now
        report = dict(self.report['diagnostics'], state=self.state, execution_enabled=self._execution_enabled(),
            last_skip=self.last_skip, failure=self.failure, region=self.region.tolist(),
            stream_available=self.arm.can_stream_joints())
        path = PoseArray()
        path.header.frame_id = 'base_link'
        path.header.stamp = self.node.get_clock().now().to_msg()
        if self.track is not None and self.track.speed_valid:
            try:
                estimate = self._estimate(self.track, now)
                report['object_velocity_m_s'] = estimate['velocity'].tolist()
                for horizon in np.linspace(0., min(8., self.duration), 25):
                    pose = Pose()
                    point = estimate['position']+estimate['velocity']*horizon
                    pose.position.x, pose.position.y = map(float, point)
                    pose.position.z = float(self.track.centroid_z)
                    pose.orientation.w = 1.
                    path.poses.append(pose)
            except ValueError:
                pass
        if not path.poses and self.track is not None:
            # A single image cannot provide velocity. Publish its observed path
            # while inference/velocity qualification run concurrently.
            report['trajectory_kind'] = 'observed_until_speed_qualified'
            for _, point in self.history.get(self.track.id, []):
                pose = Pose()
                pose.position.x, pose.position.y, pose.position.z = map(float, point)
                pose.orientation.w = 1.
                path.poses.append(pose)
        elif path.poses:
            report['trajectory_kind'] = 'qualified_constant_velocity_prediction'
        self.pub_path.publish(path)
        self.pub_preview.publish(String(data=json.dumps(report)))

    def _start_servo(self):
        self.q_command = vector(self._joints(), 7, 'servo seed').copy()
        self.q_velocity = np.zeros(7)
        self.last_tick = self._motion_time()
        self._set('FOLLOW_SYNC')

    def _remaining_follow(self, elapsed):
        if self.state == 'FOLLOW_SYNC':
            return max(0., self.sync_timeout-elapsed)+self.descend+self.close_timeout+self.lift_seconds+self.brake_seconds
        if self.state == 'FOLLOW_DESCEND':
            return max(0., self.descend-elapsed)+self.close_timeout+self.lift_seconds+self.brake_seconds
        return max(0., self.close_timeout-elapsed)+self.lift_seconds+self.brake_seconds

    def _measured(self, now):
        position, rotation = self._actual()
        tip = position+rotation.apply(self.tcp)
        return tip, rotation, self.velocity_feedback.estimate(now)

    def _matched(self, estimate, tip, tip_velocity):
        return matched(estimate['position']-estimate['velocity']*self.latency,
            estimate['covariance'], estimate['velocity'], tip, tip_velocity,
            self.plan['rotation'].apply(self.axis)[:2],
            (self.open_width-self.plan['raw_width'])/2-.002, self.belt.d_hat,
            self.sigma, self.position_tol, self.cross_tol, self.velocity_tol)

    def _stable_match(self, good, now):
        if not good:
            self.sync_since, self.sync_frames = None, 0
            return False
        if self.last_pose_stamp != self.sync_pose_stamp:
            self.sync_pose_stamp = self.last_pose_stamp
            self.sync_frames += 1
            if self.sync_since is None:
                self.sync_since = now
        return self.sync_since is not None and self.sync_frames >= 3 and now-self.sync_since >= .15

    def _grasp_verified(self, elapsed):
        width = self.grip_width
        blocked = (self._gripper_fresh() and elapsed >= .15
            and self.plan['close_width']+.003 < width < self.plan['raw_width']+.006
            and width < self.open_width-.006)
        if not blocked:
            self.grip_verified_since, self.grip_blocked_frames = None, 0
            return False
        if self.grip_counter != self.grip_verified_count:
            self.grip_verified_count = self.grip_counter
            self.grip_blocked_frames += 1
            if self.grip_verified_since is None:
                self.grip_verified_since = self._motion_time()
        return (self.grip_verified_since is not None and self.grip_blocked_frames >= 3
                and self._motion_time()-self.grip_verified_since >= .12)

    def _live_or_frozen(self, now):
        current = next((t for t in self.node._tracks if t.id == self.track.id), None)
        # Freeze only for actual disappearance/occlusion during commanded closure.
        # A present but unqualified/reacquired or inconsistent target must abort.
        occluded = (current is None or current.status == 2
                    or (current.status == 1 and now-current.stamp > .15))
        if not occluded or self.state != 'FOLLOW_CLOSE':
            tr, estimate = self._target(now, allow_lost=self.state != 'FOLLOW_SYNC')
            self.track = tr
            _, covariance = forecast_motion(
                np.r_[tr.centroid_x, tr.centroid_y, np.asarray(tr.velocity)[:2]],
                tr.motion_covariance, now+self.latency-tr.motion_stamp, tr.motion_process_noise)
            covariance[:2, :2] += self.offset_covariance
            self.frozen = dict(at=now+self.latency, position=estimate['position'].copy(),
                velocity=estimate['velocity'].copy(), covariance=covariance,
                process_noise=tr.motion_process_noise.copy(), observed_at=tr.stamp)
            return estimate
        if self.frozen is None or not 0 <= now-self.frozen['observed_at'] <= self.max_occlusion:
            raise ValueError('Closure occlusion exceeded the bounded prediction interval')
        dt = max(0., now+self.latency-self.frozen['at'])
        state, covariance = forecast_motion(
            np.r_[self.frozen['position'], self.frozen['velocity']],
            self.frozen['covariance'], dt, self.frozen['process_noise'])
        return dict(position=state[:2], velocity=state[2:],
                    covariance=covariance[:2, :2], age=now-self.frozen['observed_at'])

    def _servo(self, now, elapsed):
        phase = self.state
        clock = self._motion_time()
        dt = clock-self.last_tick
        self.last_tick = clock
        if not 0 < dt <= .06:
            raise ValueError('Follow loop missed its maximum control interval')
        tip, actual_rotation, tip_velocity = self._measured(now)
        if (not in_region(tip[:2], self.region, self.margin)
                or tip[2] < self.plan['belt_z']+.005
                or (actual_rotation.inv()*self.plan['rotation']).magnitude() > .08):
            raise ValueError('Actual TCP left the permitted region/clearance/orientation')
        if self.state in ('FOLLOW_LIFT', 'FOLLOW_BRAKE'):
            if not self._gripper_fresh() or self.grip_width <= self.plan['close_width']+.003:
                raise ValueError('Grasp retention could not be verified')
            if self.state == 'FOLLOW_LIFT':
                target_xy = self.lift_start[:2]+self.held_velocity*(elapsed+self.latency)
                z, vz = smooth_segment(self.lift_start[2], self.lift_start[2]+self.lift_offset,
                                       elapsed, self.lift_seconds)
                velocity = self.held_velocity

            else:
                delta, velocity = brake_displacement(self.held_velocity, elapsed+self.latency, self.brake_seconds)
                target_xy, z, vz = self.brake_start[:2]+delta, self.brake_start[2], 0.
            estimate = None
        else:
            if self.state == 'FOLLOW_CLOSE' and self.grip_confirmed:
                if not self._gripper_fresh() or self.grip_width <= self.plan['close_width']+.003:
                    raise ValueError('Confirmed grasp was lost during the planned close interval')
                estimate = dict(
                    position=self.grip_start[:2]+self.held_velocity*(now-self.grip_confirmed_at+self.latency),
                    velocity=self.held_velocity, covariance=np.zeros((2, 2)))
            else:
                estimate = self._live_or_frozen(now)
            window = interception_window(estimate['position'], estimate['velocity'], self.region,
                0., self._remaining_follow(elapsed), self.margin)
            if window['entry'] > 1e-6:
                raise ValueError('Object has not entered the feasible follow region')
            target_xy, velocity = estimate['position'], estimate['velocity']
            good = self._matched(estimate, tip, tip_velocity)
            if self.state == 'FOLLOW_SYNC':
                z, vz = self.plan['high_z'], 0.
                stable = self._stable_match(good and abs(tip[2]-z) < .006, now)
                # Preflight uses fixed phase durations. Reaching synchronization
                # early does not shift the descent to an unchecked upstream pose.
                if elapsed >= self.sync_timeout:
                    if not stable:
                        raise ValueError('Measured end-effector velocity did not synchronize in time')
                    self._set('FOLLOW_DESCEND')
            elif self.state == 'FOLLOW_DESCEND':
                z, vz = smooth_segment(self.plan['high_z'], self.plan['low_z'], elapsed, self.descend)
                stable = self._stable_match(good, now)
                if not good and elapsed > .2:
                    raise ValueError('Relative position/speed left capture limits during descent')
                if elapsed >= self.descend:
                    if abs(tip[2]-self.plan['low_z']) >= .006 or not stable:
                        raise ValueError('Following descent missed its verified contact time')
                    self.grip_verified_since, self.grip_verified_count, self.grip_blocked_frames = None, self.grip_counter, 0
                    self.grip_confirmed = False
                    self._grip(self.plan['close_width'], 'FOLLOW_CLOSE')
            else:
                z, vz = self.plan['low_z'], 0.
                if not good:
                    raise ValueError('Relative motion/uncertainty became unsafe during closure')
                if not self.grip_confirmed and self._grasp_verified(elapsed):
                    self.grip_confirmed = True
                    self.grip_start, self.grip_confirmed_at = tip.copy(), now
                    self.held_velocity = velocity.copy()
                # Once grip feedback confirms capture, vision can be occluded.
                # Keep following until the SAME close deadline used by preflight.
                if elapsed >= self.close_timeout:
                    if not self.grip_confirmed:
                        raise ValueError('Empty/ambiguous grasp or gripper closure timed out')
                    self.lift_start = tip.copy()
                    self._set('FOLLOW_LIFT')
        target = np.r_[target_xy, z]
        if not in_region(target_xy, self.region, self.margin):
            raise ValueError('Following reference left the grasp region')
        ws = self.node._p['workspace']
        if not (flange_in_workspace(tip-actual_rotation.apply(self.tcp), ws)
                and flange_in_workspace(target-self.plan['rotation'].apply(self.tcp), ws)):
            raise ValueError('Actual/reference flange left the configured workspace')
        joints = vector(self._joints(), 7, 'servo joint feedback')
        T = self.kin.fk(joints)
        fk_tip = T[:3, 3]+T[:3, :3]@self.tcp
        if np.linalg.norm(fk_tip-tip) > .006:
            raise ValueError('FK and measured TCP diverged during following')
        command, qd, _ = resolved_rate(self.kin, joints, self.q_command, self.q_velocity,
            self.tcp, target, self.plan['rotation'], np.r_[velocity, vz], dt, self.gain,
            self.joint_speed, self.joint_accel, self.linear_speed, self.angular_speed)
        if not flange_in_workspace(self.kin.fk(command)[:3, 3], ws):
            raise ValueError('Next joint command would leave the flange workspace')
        self.arm.stream_joints(command.tolist())
        self.q_command, self.q_velocity = command, qd
        self.report['diagnostics'].update(target_tcp_m=target.tolist(), actual_tcp_m=tip.tolist(),
            actual_tcp_velocity_m_s=tip_velocity.tolist(), reference_velocity_m_s=[*velocity.tolist(), vz],
            relative_xy_speed_m_s=float(np.linalg.norm(tip_velocity[:2]-velocity)))
        if phase == 'FOLLOW_LIFT' and elapsed >= self.lift_seconds:
            if abs(tip[2]-(self.lift_start[2]+self.lift_offset)) < .006:
                self.brake_start = target.copy()
                self._set('FOLLOW_BRAKE')
            else:
                raise ValueError('Following lift missed its verified clearance time')
        elif phase == 'FOLLOW_BRAKE' and elapsed >= self.brake_seconds:
            if self._stable_match(np.linalg.norm(tip_velocity) < .008
                    and np.linalg.norm(tip-target) < .006, now):
                # Switch to a stationary position hold; retain the gripper.
                self.arm.move_j(joints.tolist())
                self._set('FOLLOW_DONE')
            elif elapsed > self.brake_seconds+1.:
                raise ValueError('Following brake did not settle')

    def tick(self):
        if self.state in self.TERMINAL:
            return
        now = self.node.get_clock().now().nanoseconds*1e-9
        try:
            if self.motion_issued and not self._execution_enabled():
                raise ValueError('Execution disabled during active following')
            if not self._fresh():
                if self.motion_issued:
                    raise ValueError('Arm/pose/gripper feedback stale during following')
                return
            if arm_fault(self.status) or getattr(self, 'grip_fault', False):
                raise ValueError('Driver fault during following')
            self._observe(now)
            elapsed = self._motion_time()-self.started
            if self.motion_issued and not self.arm.can_stream_joints():
                raise ValueError('Exclusive continuous joint-control endpoint lost')
            if self.state == 'FOLLOW_WAIT_TARGET':
                self._acquire(now)
            elif self.state == 'FOLLOW_PLANNING' and self.planning_future.done():
                future, self.planning_future = self.planning_future, None
                try:
                    self.plan = future.result()
                    if self.plan['sync_at']-now < self.prepare+self.approach:
                        raise ValueError('Object arrived too late after path preflight')
                    self._ready_plan(now)
                    self._save_plan()
                    self._set('FOLLOW_READY')
                except ValueError as error:
                    self._skip(error, now)
            elif self.state == 'FOLLOW_READY':
                try:
                    self._ready_plan(now)
                    self._request(now)
                    remaining = self.plan['sync_at']-now
                    if remaining < self.prepare+self.approach:
                        raise ValueError('Follow startup window passed')
                    if not self._execution_enabled():
                        self.report['diagnostics']['preview_would_start'] = True
                    elif remaining <= self.prepare+self.approach+.15:
                        if not self.arm.can_stream_joints():
                            self.report['diagnostics']['blocked_reason'] = 'Exclusive /control/move_js unavailable'
                            return
                        if not self.enable.service_is_ready():
                            raise ValueError('Arm enable service unavailable')
                        request = SetBool.Request(data=True)
                        self.enable_future = self.enable.call_async(request)
                        self.motion_issued = True
                        self.startup_at = self._motion_time()
                        self._set('FOLLOW_ENABLE')
                except ValueError as error:
                    self._skip(error, now)
            elif self.state in ('FOLLOW_ENABLE', 'FOLLOW_OPEN', 'FOLLOW_APPROACH', 'FOLLOW_WAIT_SYNC'):
                self._ready_plan(now, before_motion=False)
                self._request(now)
                if self._motion_time()-self.startup_at > self.prepare+self.approach+self.sync_timeout:
                    raise ValueError('Following startup exceeded its budget')
                if self.state == 'FOLLOW_ENABLE':
                    if self.enable_future.done():
                        if not self.enable_future.result().success:
                            raise ValueError('Arm enable failed')
                        self._grip(self.open_width, 'FOLLOW_OPEN')
                    elif elapsed > self.prepare:
                        raise ValueError('Arm enabling exceeded preparation budget')
                elif self.state == 'FOLLOW_OPEN':
                    if self.grip_width >= self.open_width-.003:
                        self._ready_plan(now)
                        if self.plan['sync_at']-now < self.approach:
                            raise ValueError('Insufficient approach time after enabling/opening')
                        self._motion(self.plan['above'], 'FOLLOW_APPROACH', self.plan['joints'][0])
                    elif elapsed > self.prepare:
                        raise ValueError('Gripper opening exceeded preparation budget')
                elif self.state == 'FOLLOW_APPROACH':
                    if now > self.plan['sync_at']+.15:
                        raise ValueError('Arm missed the planned synchronization time')
                    if self._arrived():
                        self._set('FOLLOW_WAIT_SYNC')
                elif now >= self.plan['sync_at']:
                    tr, estimate = self._target(now)
                    if not in_region(estimate['position'], self.region, self.margin):
                        raise ValueError('Object has not entered the grasp region')
                    if np.linalg.norm(estimate['position']-self.plan['above'][:2]) > .02:
                        raise ValueError('Object missed the verified synchronization point')
                    self.track = tr
                    self._start_servo()
            elif self.state in self.SERVO_STATES:
                if self.state in ('FOLLOW_SYNC', 'FOLLOW_DESCEND'):
                    self._request(now)
                self._servo(now, elapsed)
        except Exception as error:
            self._abort(error)
        finally:
            self._preview(now)
