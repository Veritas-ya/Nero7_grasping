"""One stationary pick/return cycle inside the formal GR-ConvNet controller.

Inference supplies the point, yaw and closing width. No geometry fallback is
used. Preview is the default; execution_enabled starts exactly one cycle.
"""
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import yaml
import rclpy
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import PoseStamped
from agx_arm_msgs.msg import AgxArmStatus, GripperStatus
from std_srvs.srv import SetBool
from ament_index_python.packages import get_package_share_directory
from .kinematics import Kinematics, yaw_orientations
from .gripper_feedback import GripperFeedbackMonitor
from .fixed_bin import (load_fixed_bin, plan_fixed_bin, plan_fixed_bin_return,
                        placement_summary, fixed_bin_profile_current)
from .ready_pose import require_limits


class StaticOnce:
    def __init__(self, node):
        self.node = node
        self.arm = node._arm
        self.use_sim_time = bool(node.get_parameter('use_sim_time').value)
        self.state = 'STATIC_INFER'
        self.started = self._motion_time()
        self.pose = None
        self.pose_time = 0
        self.status = None
        self.status_time = 0
        self.goal = None
        self.settled = None
        self.plan = None
        self.track = None
        self.motion_issued = False
        self.grip_goal = None
        self.gripper_feedback = GripperFeedbackMonitor()
        self.failure = None
        self.bin_planner = ThreadPoolExecutor(max_workers=1, thread_name_prefix='bin_preflight')
        self.bin_plan_future = None
        declare = lambda key, default: node.declare_parameter(key, default).value
        path = declare('calibration_file', '/home/ubuntu22/Nero7_grasping/src/nero7_bringup/config/calibration.yaml')
        with open(path) as f:
            self.calibration = yaml.safe_load(f)
        self.tcp = np.asarray(self.calibration['gripper_tcp'], float)
        self.axis = declare('closing_axis_flange', [0., 0., 1.])
        self.roi = declare('static_roi_px', [180, 355, 1060, 475])
        self.center_source = declare('static_center_source', 'network')
        if self.center_source not in ('network', 'track_centroid'):
            raise ValueError('static_center_source must be network or track_centroid')
        self.max_network_track_xy = float(declare('static_max_network_track_xy_m', .04))
        if not 0 < self.max_network_track_xy <= .065:
            raise ValueError('static_max_network_track_xy_m must be within (0, 0.065]')
        self.offset = float(declare('static_descend_offset', .018))
        self.clearance = float(declare('static_belt_clearance', .012))
        self.open_width = float(declare('static_open_width', .070))
        self.infer_timeout = float(declare('static_infer_timeout', 30.))
        self.plan_file = declare('static_plan_file', '/tmp/nero7_formal_static_plan.json')

        self.bin_file = declare('static_bin_file', str(
            Path(get_package_share_directory('nero7_control'))/'config/fixed_bin.yaml'))
        self.bin_profile = load_fixed_bin(self.bin_file)
        self.bin_workspace = {key: float(declare('static_bin_workspace.'+key, value))
                              for key, value in dict(x_min=-.50, x_max=.45,
                                  y_min=-.46, y_max=.35, z_min=-.05).items()}

        urdf = declare('urdf_path', '/home/ubuntu22/nero_ws/install/nero_description/share/nero_description/urdf/nero_description.urdf')
        self.kin = Kinematics(urdf)
        prefix = node._p['topic_prefix']
        node.create_subscription(PoseStamped, prefix+'feedback/tcp_pose', self._pose_cb, 5)
        node.create_subscription(AgxArmStatus, prefix+'feedback/arm_status', self._status_cb, 5)
        node.create_subscription(GripperStatus, prefix+'feedback/gripper_status', self._gripper_cb, 5)
        self.enable = node.create_client(SetBool, prefix+'enable_agx_arm')

    def _pose_cb(self, msg):
        self.pose = msg.pose
        self.pose_time = time.monotonic()

    def _status_cb(self, msg):
        self.status = msg
        self.status_time = time.monotonic()

    def _gripper_cb(self, msg):
        stamp = msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        now = self.node.get_clock().now().nanoseconds*1e-9
        if not 0 <= now-stamp <= .3:
            return
        fault = any(getattr(msg, key) for key in (
            'voltage_too_low', 'motor_overheating', 'driver_overcurrent',
            'driver_overheating', 'sensor_status', 'driver_error_status'))
        self.gripper_feedback.update(msg.width, stamp, time.monotonic(),
                                     msg.driver_enable_status, fault)

    def _actual(self):
        p, q = self.pose.position, self.pose.orientation
        return np.array([p.x, p.y, p.z]), Rotation.from_quat([q.x, q.y, q.z, q.w])

    def _motion_time(self):
        """Use Gazebo's clock for motion durations; hardware keeps monotonic time."""
        if getattr(self, 'use_sim_time', False):
            return self.node.get_clock().now().nanoseconds*1e-9
        return time.monotonic()

    def _set(self, state):
        self.state = state
        self.started = self._motion_time()
        self.node.get_logger().info(state)

    def _abort(self, reason):
        self.failure = reason
        self.node.get_logger().error('STATIC_ABORT: '+reason)
        if getattr(self, 'bin_plan_future', None) is not None:
            self.bin_plan_future.cancel()
        if self.motion_issued:
            self.arm.estop()
        self._set('STATIC_ABORT')

    def shutdown(self):
        planner = getattr(self, 'bin_planner', None)
        if planner is not None:
            planner.shutdown(wait=False, cancel_futures=True)
        if self.motion_issued:
            future = self.arm.estop()
            if future is not None and self.node.context.ok():
                rclpy.spin_until_future_complete(self.node, future, timeout_sec=.5)
                if not future.done():
                    self.node.get_logger().error('Emergency stop acknowledgement timed out')

    def _fresh(self):
        now = time.monotonic()
        return (self.pose is not None and self.status is not None
                and now-self.pose_time < .3 and now-self.status_time < .3
                and (self.node.get_clock().now()-self.node._feedback_time).nanoseconds*1e-9 < .3)

    def _motion(self, point, state, joints=None, rotation=None):
        rotation = self.plan['rotation'] if rotation is None else rotation
        flange = np.asarray(point)-rotation.apply(self.tcp)
        ws = self.bin_workspace if state.startswith('STATIC_BIN_') else self.node._p['workspace']
        if not (ws['x_min'] <= flange[0] <= ws['x_max'] and ws['y_min'] <= flange[1] <= ws['y_max'] and flange[2] >= ws['z_min']):
            raise ValueError('Flange pose outside configured workspace')
        pose = [*flange.tolist(), *rotation.as_euler('xyz').tolist()]
        self.goal = (flange, rotation)
        self.settled = None
        self.motion_issued = True
        if joints is None:
            self.arm.move_l(pose)
        else:
            self.arm.move_j(joints.tolist())
        self._set(state)

    def _measured_bin_seed(self, require_motion_status=True):
        actual = self.arm.get_joint_state()
        if actual is None or len(actual)<7:
            raise ValueError('Missing bin path joint feedback')
        q = require_limits(self.kin, actual[:7]).copy()
        if require_motion_status and self.status.motion_status != 0:
            raise ValueError('Arm must be settled before starting the next bin segment')
        position, rotation = self._actual()
        transform = self.kin.fk(q)
        if (np.linalg.norm(transform[:3, 3]-position)>.006
                or (Rotation.from_matrix(transform[:3, :3]).inv()*rotation).magnitude()>.06):
            raise ValueError('Bin path FK and measured flange disagree')
        return q

    def _require_bin_seed(self, expected):
        q = self._measured_bin_seed()
        delta = np.max(np.abs(q-expected))
        if delta>.05:
            raise ValueError('Arm changed after bin preflight: max joint delta %.2f deg'
                             % np.rad2deg(delta))

    def _require_bin_pose(self, tip, rotation, require_motion_status=True):
        q = self._measured_bin_seed(require_motion_status=require_motion_status)
        actual, actual_rotation = self._actual()
        actual_tip = actual+actual_rotation.apply(self.tcp)
        if (np.linalg.norm(actual_tip-tip)>.006
                or (rotation.inv()*actual_rotation).magnitude()>.06):
            raise ValueError('Measured TCP has not reached the bin segment endpoint')
        return q

    def _begin_bin_replan(self, returning=False):
        if returning:
            placement = self.plan['placement']
            q = self._require_bin_pose(placement['above'], placement['rotation'])
            function, args = plan_fixed_bin_return, (placement,)
            state = 'STATIC_BIN_RETURN_REPLAN'
        else:
            self._require_bin_holding()
            q = self._require_bin_pose(self.plan['lift'], self.plan['rotation'])
            function, args = plan_fixed_bin, (self.bin_profile,)
            state = 'STATIC_BIN_REPLAN'
        self.bin_plan_seed = q.copy()
        # Keep ROS callbacks running while IK and sampled route checks execute.
        self.bin_plan_future = self.bin_planner.submit(function, self.kin, q,
            self.tcp.copy(), *args, self.plan['belt_z'], dict(self.bin_workspace))
        self._set(state)

    def _finish_bin_replan(self, returning=False):
        if not self.bin_plan_future.done():
            if self._motion_time()-self.started > 20:
                raise ValueError('Bin path recomputation timed out; retaining current pose')
            return
        placement = self.bin_plan_future.result()
        # A new plan is valid only if the measured arm stayed at its snapshot.
        self._require_bin_seed(self.bin_plan_seed)
        self.plan['placement'] = placement
        self.bin_plan_future = None
        self.node.get_logger().info('BIN_PATH_REPLANNED '+json.dumps(dict(
            stage='return' if returning else 'transfer',
            actual_seed_joints_rad=self.bin_plan_seed.tolist(),
            fixed_bin=placement_summary(placement))))
        if returning:
            self._bin_return(0)
        else:
            self._bin_transfer(0)

    def _require_bin_holding(self):
        width = self.gripper_feedback.reading(time.monotonic())
        if width is None or not self.plan['close_width']+.003 < width < self.open_width-.01:
            raise ValueError('Fresh feedback no longer confirms a held object; bin path stopped')

    def _bin_transfer(self, index):
        placement = self.plan['placement']
        self._require_bin_holding()
        previous = placement['lift_joints'] if index==0 else placement['transfer_joints'][index-1]
        self._require_bin_seed(previous)
        q = placement['transfer_joints'][index]
        transform = self.kin.fk(q)
        rotation = Rotation.from_matrix(transform[:3, :3])
        self.bin_transfer_index = index
        self._motion(transform[:3, 3]+rotation.apply(self.tcp), 'STATIC_BIN_TRANSFER', q, rotation)

    def _bin_return(self, index):
        placement = self.plan['placement']
        previous = placement['above_joints'] if index==0 else placement['return_joints'][index-1]
        self._require_bin_seed(previous)
        q = placement['return_joints'][index]
        transform = self.kin.fk(q)
        rotation = Rotation.from_matrix(transform[:3, :3])
        self.bin_return_index = index
        self._motion(transform[:3, 3]+rotation.apply(self.tcp), 'STATIC_BIN_RETURN', q, rotation)

    def _arrived(self):
        position, rotation = self._actual()
        error = np.linalg.norm(position-self.goal[0])
        angle = (rotation.inv()*self.goal[1]).magnitude()
        if self._motion_time()-self.started > 20:
            raise ValueError('Motion timed out; position error %.3f m' % error)
        if error < .006 and angle < .06 and self.status.motion_status == 0 and self._motion_time()-self.started > .3:
            if self.settled is None:
                self.settled = self._motion_time()
            return self._motion_time()-self.settled > .3
        self.settled = None
        return False

    def _grip(self, width, state):
        self.grip_goal = width
        self.gripper_feedback.begin(width, time.monotonic())
        self.motion_issued = True
        self.arm.move_gripper(width, force=1.5 if state.endswith('_CLOSE') else 1.)
        self._set(state)

    def _make_plan(self, result, track, approach_height=.12, source='GR-ConvNet'):
        """Preflight the pickup only; plan placement from measured joints after lift."""
        center = np.asarray(result['center'], float)
        network_track_xy = float(np.linalg.norm(
            center[:2]-[track.centroid_x, track.centroid_y]))
        if (not np.isfinite(center).all()
                or network_track_xy > getattr(self, 'max_network_track_xy', .04)):
            raise ValueError('Network grasp point outside target neighborhood')
        if not track.belt_z+.005 < center[2] < track.belt_z+.12:
            raise ValueError('Network point depth inconsistent with belt')
        network_center = center.copy()
        if source == 'GR-ConvNet' and getattr(self, 'center_source', 'network') == 'track_centroid':
            center = np.asarray([track.centroid_x, track.centroid_y, track.centroid_z], float)
            if not np.isfinite(center).all() or not track.belt_z+.005 < center[2] < track.belt_z+.12:
                raise ValueError('Track centroid depth inconsistent with belt')
            source = 'RGB-D track centroid + GR-ConvNet yaw/width'
        raw_width = float(result['raw_width_mm'])/1000
        if not .012 < raw_width < self.open_width-.002:
            raise ValueError('Network grasp width outside usable opening')
        close_width = float(result['width_mm'])/1000
        if not np.isfinite([close_width, result['yaw']]).all() or not 0 < close_width < raw_width:
            raise ValueError('Network closing width or direction invalid')
        above = center.copy()
        above[2] += approach_height
        grasp = center.copy()
        grasp[2] = max(center[2]-self.offset, track.belt_z+self.clearance)
        lift = grasp.copy()
        # Pickup lift is deliberately independent of the material-bin geometry.
        # Any additional carry-height raise is planned from measured joints only
        # after the object has been grasped and this lift has completed.
        lift[2] = grasp[2]+.10
        joints = self.arm.get_joint_state()
        if joints is None:
            raise ValueError('Missing joint feedback')
        seed = np.asarray(joints[:7], float)
        actual, actual_rotation = self._actual()
        if np.linalg.norm(self.kin.fk(seed)[:3, 3]-actual) > .003:
            raise ValueError('URDF and flange feedback do not agree')
        options, rejections = [], []
        for orientation_index, rotation in enumerate(
                yaw_orientations(self.calibration['grasp_rpy'], self.axis, result['yaw'])):
            previous = seed
            solutions = []
            try:
                for stage, point in [('above', above), ('grasp', grasp), ('lift', lift)]:
                    flange = point-rotation.apply(self.tcp)
                    ws = self.node._p['workspace']
                    if not (ws['x_min'] <= flange[0] <= ws['x_max']
                            and ws['y_min'] <= flange[1] <= ws['y_max']
                            and flange[2] >= ws['z_min']):
                        raise ValueError(stage+' flange outside pickup workspace')
                    previous = self.kin.solve(flange, rotation, previous)
                    solutions.append(previous)
            except ValueError as error:
                rejections.append('orientation %d %s: %s' %
                                  (orientation_index+1, stage, error))
                continue
            approach_step = float(np.max(np.abs(solutions[0]-seed)))
            if approach_step > 2.3:
                rejections.append('orientation %d approach joint step %.3f rad exceeds 2.3 rad'
                                  % (orientation_index+1, approach_step))
                continue
            clear = True
            minimum_tip_z = float('inf')
            for alpha in np.linspace(0, 1, 150):
                T = self.kin.fk(seed*(1-alpha)+solutions[0]*alpha)
                tip = T[:3, 3]+T[:3, :3]@self.tcp
                minimum_tip_z = min(minimum_tip_z, float(tip[2]))
                if tip[2] < track.belt_z+.05:
                    clear = False
                    break
            if clear:
                options.append((np.linalg.norm(solutions[0]-seed), rotation, solutions))
            else:
                rejections.append('orientation %d approach tip z %.3f m below %.3f m'
                                  % (orientation_index+1, minimum_tip_z, track.belt_z+.05))
        if not options:
            detail = '; '.join(rejections) if rejections else 'no orientation generated'
            raise ValueError('No verified pickup path for network candidate '
                             '(above/grasp/lift): '+detail)
        _, rotation, solutions = min(options, key=lambda option: option[0])
        plan = dict(center=center, above=above, grasp=grasp, lift=lift,
                    rotation=rotation, joints=solutions, raw_width=raw_width,
                    close_width=close_width,
                    seed_joints=seed.copy(), seed_flange=actual.copy(),
                    seed_rotation=actual_rotation, placement=None, belt_z=float(track.belt_z))
        output = dict(source=source, score=result['score'],
                      center_source=getattr(self, 'center_source', 'network'),
                      network_track_xy_delta_mm=network_track_xy*1000.,
                      network_center_base=network_center.tolist(),
                      center_base=center.tolist(), fingertip_grasp=grasp.tolist(),
                      yaw=result['yaw'], flange_rpy=rotation.as_euler('xyz').tolist(),
                      raw_width_mm=result['raw_width_mm'], close_width_mm=result['width_mm'],
                      approach_joints=solutions[0].tolist(), track_id=track.id,
                      generated_at_unix=time.time(), fixed_bin=None,
                      placement_status='deferred_until_after_lift')
        with open(self.plan_file, 'w') as f:
            json.dump(output, f, indent=2)
        self.node.get_logger().info('NETWORK_STATIC_PLAN '+json.dumps(output))
        return plan

    def tick(self):
        if self.state in ('STATIC_DONE', 'STATIC_ABORT'):
            return
        try:
            if self.plan is not None and not fixed_bin_profile_current(self.bin_profile):
                raise ValueError('Fixed bin profile changed; recompute the complete path')
            if not self._fresh():
                if self.motion_issued:
                    self._abort('Feedback stale')
                elif self._motion_time()-self.started > self.infer_timeout:
                    self._abort('Feedback unavailable')
                return
            if self.status.err_status or self.status.arm_status in (1, 5, 7, 9, 10):
                self._abort('Driver fault '+str(self.status.arm_status))
                return
            now = self.node.get_clock().now()
            stamp = now.nanoseconds*1e-9
            if self.state in ('STATIC_INFER', 'STATIC_READY'):
                u0, v0, u1, v1 = self.roi
                tracks = [t for t in self.node._tracks if t.status == 1 and stamp-t.stamp < .2
                          and u0 <= t.u_px <= u1 and v0 <= t.v_px <= v1
                          and np.linalg.norm(t.velocity) < .015
                          and -.45 < t.centroid_x < .45 and -.35 < t.centroid_y < .35]
                if not tracks:
                    if self._motion_time()-self.started > self.infer_timeout:
                        self._abort('No stationary confirmed target')
                    return
                track = min(tracks, key=lambda t: abs(t.centroid_x+.36)+abs(t.centroid_y-.20))
                self.node._fsm.target_id = track.id
                self.node._fsm.state = 'TRACK'
                self.node._maybe_request_inference(now)
                if self.node._inference is None:
                    self._abort('GR-ConvNet model unavailable; no fallback allowed')
                    return
                result = self.node._inference.latest(max_age=self.node._p['result_max_age'])
                if result is not None and result.get('status') != 0:
                    self.node.get_logger().warning('Network has no usable grasp: '+str(result), throttle_duration_sec=5.)
                if result is None or result.get('status') != 0 or abs(result['stamp']-track.stamp) > .3:
                    if self._motion_time()-self.started > self.infer_timeout:
                        self._abort('No valid fresh network grasp')
                    return
                if self.plan is None:
                    try:
                        self.plan = self._make_plan(result, track)
                    except ValueError as e:
                        self.node.get_logger().warning('Rejected network candidate: '+str(e), throttle_duration_sec=5.)
                        if self._motion_time()-self.started > self.infer_timeout:
                            self._abort(str(e))
                        return
                    self.track = track
                    self.node._publish_candidate(result)
                    self._set('STATIC_READY')
                if not self.node.get_parameter('execution_enabled').value:
                    return
                if np.linalg.norm(np.array([track.centroid_x, track.centroid_y])-
                                  [self.track.centroid_x, self.track.centroid_y]) > .008:
                    self.plan = None
                    self._set('STATIC_INFER')
                    return
                if not self.enable.service_is_ready():
                    raise ValueError('Enable service unavailable')
                request = SetBool.Request()
                request.data = True
                self.enable_future = self.enable.call_async(request)
                self._set('STATIC_ENABLE')
            elif self.state == 'STATIC_ENABLE':
                if self.enable_future.done():
                    if not self.enable_future.result().success:
                        raise ValueError('Arm enable failed')
                    self._grip(self.open_width, 'STATIC_OPEN')
                elif self._motion_time()-self.started > 8:
                    raise ValueError('Enable timed out')
            elif self.state == 'STATIC_OPEN':
                width = self.arm.get_gripper_width()
                self.node.get_logger().info('STATIC_OPEN current_width='+str(width), throttle_duration_sec=1.0)
                if self.gripper_feedback.opening_confirmed(time.monotonic()):
                    self._motion(self.plan['above'], 'STATIC_APPROACH', self.plan['joints'][0])
                elif self._motion_time()-self.started > 4.:
                    raise ValueError('Gripper opening not confirmed: goal=%.1f mm, reported=%s m; arm approach blocked'
                                     % (self.open_width*1000, width))
            elif self.state == 'STATIC_APPROACH' and self._arrived():
                self._motion(self.plan['grasp'], 'STATIC_DESCEND')
            elif self.state == 'STATIC_DESCEND' and self._arrived():
                self._grip(self.plan['close_width'], 'STATIC_CLOSE')
            elif self.state == 'STATIC_CLOSE' and self._motion_time()-self.started > 1.5:
                width = self.gripper_feedback.reading(time.monotonic())
                if width is None:
                    if self._motion_time()-self.started > 4.:
                        raise ValueError('Fresh gripper feedback unavailable after closing')
                    return
                self.node.get_logger().info('STATIC_CLOSE width='+str(width))
                if width <= self.plan['close_width']+.003 or width >= self.open_width-.01:
                    self.failure = 'Empty grasp'
                    self._grip(self.open_width, 'STATIC_RECOVER_OPEN')
                else:
                    self._motion(self.plan['lift'], 'STATIC_LIFT')
            elif self.state == 'STATIC_LIFT' and self._arrived():
                self._set('STATIC_HOLD')
            elif self.state == 'STATIC_HOLD' and self._motion_time()-self.started > 2:
                self._begin_bin_replan()
            elif self.state == 'STATIC_BIN_REPLAN':
                self._require_bin_holding()
                self._finish_bin_replan()
            elif self.state == 'STATIC_BIN_TRANSFER':
                self._require_bin_holding()
                if self._arrived():
                    placement = self.plan['placement']
                    if self.bin_transfer_index+1 < len(placement['transfer_joints']):
                        self._bin_transfer(self.bin_transfer_index+1)
                    else:
                        self._require_bin_seed(placement['above_joints'])
                        self._motion(placement['release'], 'STATIC_BIN_DESCEND', rotation=placement['rotation'])
            elif self.state == 'STATIC_BIN_DESCEND':
                self._require_bin_holding()
                if self._arrived():
                    self._require_bin_pose(self.plan['placement']['release'], self.plan['placement']['rotation'])
                    self._grip(self.open_width, 'STATIC_BIN_RELEASE')
            elif self.state == 'STATIC_BIN_RELEASE' and self._motion_time()-self.started > 1.5:
                if self.gripper_feedback.opening_confirmed(time.monotonic()):
                    placement = self.plan['placement']
                    # Nero reports the gripper's not-exactly-at-target state in
                    # the shared arm motion_status.  The arm was settled and
                    # pose-verified before opening; after opening, verify fresh
                    # joints, FK and TCP again without using that shared bit.
                    self._require_bin_pose(placement['release'], placement['rotation'],
                                           require_motion_status=False)
                    self._motion(placement['above'], 'STATIC_BIN_RETRACT', rotation=placement['rotation'])
                elif self._motion_time()-self.started > 4.:
                    raise ValueError('Bin release opening not confirmed')
            elif self.state == 'STATIC_BIN_RETRACT' and self._arrived():
                if self.plan['placement']['return_joints']:
                    self._begin_bin_replan(returning=True)
                else:
                    self._set('STATIC_DONE')
            elif self.state == 'STATIC_BIN_RETURN_REPLAN':
                width = self.gripper_feedback.reading(time.monotonic())
                if width is None or width < self.open_width-.003:
                    raise ValueError('Fresh open-gripper feedback required before replanning return')
                self._finish_bin_replan(returning=True)
            elif self.state == 'STATIC_BIN_RETURN':
                width = self.gripper_feedback.reading(time.monotonic())
                if width is None or width < self.open_width-.003:
                    raise ValueError('Fresh open-gripper feedback required during bin return')
                if self._arrived():
                    points = self.plan['placement']['return_joints']
                    self._require_bin_seed(points[self.bin_return_index])
                    if self.bin_return_index+1 < len(points):
                        self._bin_return(self.bin_return_index+1)
                    else:
                        self._set('STATIC_INITIAL_READY')
            elif self.state == 'STATIC_INITIAL_READY':
                self._set('STATIC_DONE')
            elif self.state == 'STATIC_RECOVER_OPEN' and self._motion_time()-self.started > 1.5:
                if not self.gripper_feedback.opening_confirmed(time.monotonic()):
                    if self._motion_time()-self.started > 4.:
                        raise ValueError('Gripper release opening not confirmed')
                    return
                if self.failure:
                    self._motion(self.plan['above'], 'STATIC_RECOVER_RAISE')
                else:
                    self._set('STATIC_DONE')
            elif self.state in ('STATIC_RETRACT', 'STATIC_RECOVER_RAISE') and self._arrived():
                self._set('STATIC_ABORT' if self.failure else 'STATIC_DONE')
        except Exception as error:
            self._abort(str(error))
