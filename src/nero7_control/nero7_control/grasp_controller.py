"""Grasp controller node: FSM glue, arm command execution, safety checks.

Subscribes block tracks, color + aligned depth + camera info for inference
requests, arm feedback for motion/gripper state and the feedback watchdog.
Runs the GraspFsm at fsm_rate and executes the returned commands through the
configured ArmClient backend.
"""
import math
import signal

import cv2
import numpy as np
import rclpy
from rclpy.signals import SignalHandlerOptions
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, CameraInfo, JointState
from std_srvs.srv import SetBool
from std_msgs.msg import String, Float64
from cv_bridge import CvBridge
import tf2_ros
from tf2_ros import TransformException
import yaml

from nero7_msgs.msg import BlockTrackArray, GraspCandidate

from .arm_client import RosTopicArmClient, SdkArmClient
from .grasp_inference import GraspInference
from .fsm import GraspFsm, HALT
from ament_index_python.packages import get_package_share_directory
from pathlib import Path
from .belt import BeltModel
from .visualization import SpeedImageViewer
from agx_arm_msgs.msg import AgxArmStatus


class TrackView:
    """Msg-like track for the FSM."""

    def __init__(self, msg):
        self.id = msg.id
        self.s = msg.s
        self.speed = msg.speed
        self.status = msg.status
        self.height = msg.height
        self.belt_z = msg.belt_z
        self.centroid_x = msg.centroid_base.x
        self.centroid_y = msg.centroid_base.y
        self.u_px = msg.u_px
        self.v_px = msg.v_px
        self.centroid_z = msg.centroid_base.z
        self.velocity = [msg.velocity_base.x, msg.velocity_base.y, msg.velocity_base.z]
        self.stamp = msg.stamp.sec + msg.stamp.nanosec * 1e-9
        self.motion_stamp = msg.motion_stamp.sec+msg.motion_stamp.nanosec*1e-9
        self.speed_valid = msg.speed_valid
        self.motion_covariance = np.asarray(msg.motion_covariance, float).reshape(4, 4)
        self.motion_process_noise = np.asarray(msg.motion_process_noise, float).reshape(2, 2)
        self.observed_position = np.array([msg.observed_centroid_base.x, msg.observed_centroid_base.y,
                                           msg.observed_centroid_base.z])

    @property
    def speed_ok(self):
        return self.speed_valid


class GraspControllerNode(Node):
    def __init__(self):
        super().__init__('grasp_controller')
        p = self.declare_parameter('params_file', '').value
        file_params = {}
        if p:
            with open(p) as f:
                file_params = yaml.safe_load(f)['grasp_controller']['ros__parameters']
        params = {}
        # Declare file values as defaults so explicit ROS overrides still win.
        for name, default in _DEFAULTS.items():
            value = file_params.get(name, default)
            if name in ('calibration_file', 'belt_file') and not value:
                filename = 'calibration.yaml' if name == 'calibration_file' else 'belt.yaml'
                value = str(Path(get_package_share_directory('nero7_bringup'))/'config'/filename)
            if isinstance(default, dict):
                values = dict(default, **value)
                params[name] = {key: self.declare_parameter(
                    name+'.'+key, setting).value for key, setting in values.items()}
            else:
                params[name] = self.declare_parameter(name, value).value
        self._p = params
        self._mode = self.declare_parameter('mode', 'conveyor').value
        self.declare_parameter('execution_enabled', False)
        self.declare_parameter('exit_on_complete', False)
        self._static_once = None
        self._continuous_motion_issued = False
        # launch-level dry_run override (string from launch) beats the yaml
        dry_run = (self.get_parameter('dry_run').value if self.has_parameter('dry_run')
                   else self.declare_parameter('dry_run', '').value)
        if dry_run in ('true', 'false'):
            params['dry_run'] = (dry_run == 'true')
        elif isinstance(dry_run, bool):
            params['dry_run'] = dry_run

        self._bridge = CvBridge()
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        qos = QoSProfile(depth=1)
        if params['qos_reliability'] == 'best_effort':
            qos.reliability = ReliabilityPolicy.BEST_EFFORT

        # --- inputs ---
        self._tracks = []
        self._tracks_stamp = None
        self._color = None
        self._depth = None
        self._depth_stamp = None
        self._color_stamp = None
        self._K = None
        self._color_frame = self._depth_frame = None
        self._color_info = self._depth_info = None
        self.create_subscription(BlockTrackArray, params['tracks_topic'],
                                 self._tracks_cb, qos)
        self.create_subscription(Image, params['color_topic'], self._color_cb, qos)
        self.create_subscription(Image, params['depth_topic'], self._depth_cb, qos)
        self.create_subscription(CameraInfo, params['color_camera_info'],
                                 self._info_cb, qos)
        self.create_subscription(CameraInfo, params['depth_camera_info'], self._depth_info_cb, qos)

        # --- outputs ---
        self._pub_state = self.create_publisher(String, '/grasp/state', 5)
        self._pub_candidate = self.create_publisher(
            GraspCandidate, '/grasp/candidate', 5)
        self._pub_delta_s = self.create_publisher(Float64, '/grasp/delta_s', 5)
        self._srv_reset = self.create_service(SetBool, '/grasp/reset', self._reset_cb)

        # --- belt + fsm ---
        belt = {}
        if params['belt_file']:
            try:
                with open(params['belt_file']) as f:
                    belt = yaml.safe_load(f) or {}
            except OSError as e:
                self.get_logger().warn(f'cannot read belt file: {e}')
        self._belt = BeltModel.from_yaml(belt)
        with open(params['calibration_file']) as f:
            self._calibration = yaml.safe_load(f)
        self._fsm = GraspFsm(self._belt, self._fsm_cfg(params))

        # --- arm ---
        backend = params['backend']
        if backend == 'sdk':
            self._arm = SdkArmClient()
            self._arm.set_limits(params['arm_speed_percent'])
        else:
            self._arm = RosTopicArmClient(self, prefix=params['topic_prefix'])
        self._feedback_time = self.get_clock().now()
        self._arm_status = None
        self._arm_status_time = self._feedback_time
        self.create_subscription(JointState, params['topic_prefix'] + 'feedback/joint_states',
                                 self._feedback_cb, 5)
        self.create_subscription(AgxArmStatus, params['topic_prefix']+'feedback/arm_status',
                                 self._arm_status_cb, 5)

        # --- inference ---
        self._inference = None
        try:
            self._inference = GraspInference(
                params['model_path'], device=params['inference_device'],
                max_rate=params['infer_max_rate'],
                q_threshold=params.get('q_threshold', 0.25),
                target_radius_px=params['infer_target_radius_px'])
        except Exception as e:  # noqa: BLE001 - run without the network
            self.get_logger().error(f'failed to load grasp model: {e}')

        self._last_cmd_pose = None
        self._watchdog_tripped = False
        self._last_pub_delta = None
        if self._mode == 'conveyor_once':
            from .conveyor_once import ConveyorOnce
            self._static_once = ConveyorOnce(self)
        elif self._mode == 'static_once':
            from .static_once import StaticOnce
            self._static_once = StaticOnce(self)
        elif self._mode != 'conveyor':
            raise ValueError('mode must be conveyor, static_once or conveyor_once')
        # runtime-tunable lead (used by lead_calib): declared param wins over yaml
        k_lead = (self.get_parameter('k_lead').value if self.has_parameter('k_lead')
                  else self.declare_parameter('k_lead', params.get('k_lead', 0.05)).value)
        params['k_lead'] = float(k_lead)
        self._k_lead_sync = self.create_timer(1.0, self._sync_k_lead)
        self._state_msg_timer = self.create_timer(1.0, self._publish_state)
        self._timer = self.create_timer(1.0 / params['fsm_rate'], self._tick)
        self._speed_viewer = None
        if params['visualization_enabled']:
            self._speed_viewer = SpeedImageViewer(self.get_logger(), params['visualization_topic'])
            self._viewer_timer = self.create_timer(1.0, self._speed_viewer.check)
        self.get_logger().info('grasp_controller started')

    def destroy_node(self):
        if getattr(self, '_speed_viewer', None) is not None:
            self._speed_viewer.close()
        return super().destroy_node()

    def _sync_k_lead(self):
        """Pick up `ros2 param set /grasp_controller k_lead <v>` at runtime."""
        v = self.get_parameter('k_lead').value
        if abs(float(v) - self._fsm.cfg['k_lead']) > 1e-6:
            self._fsm.cfg['k_lead'] = float(v)
            self.get_logger().info(f'k_lead updated to {v}')

    def _stop_continuous(self, reason):
        self._watchdog_tripped = True
        self._fsm._enter(HALT)
        self.get_logger().error(reason)
        if self._continuous_motion_issued:
            return self._arm.estop()
        return None

    # ---------------- config ----------------

    def _fsm_cfg(self, params):
        keys = ('approach_height', 'descend_offset', 'lift_height', 't_approach',
                'tau_desc', 'k_lead', 'gate_tol', 'abort_tol', 'yaw_off',
                'grasp_roll', 'grasp_pitch', 'fallback_width_mm', 'gripper_force',
                'speed_min', 'speed_max', 'timeout_approach', 'timeout_wait',
                'timeout_descend', 'timeout_grasp', 'timeout_lift',
                'timeout_place', 'miss_halt_threshold', 'dry_run',
                'bin_pre_joints', 'bin_drop_pose')
        cfg = {k: params[k] for k in keys if params.get(k) is not None}
        cfg['gripper_tcp'] = self._calibration['gripper_tcp']
        cfg['grasp_rpy'] = self._calibration['grasp_rpy']
        cfg['closing_axis_flange'] = [0., 0., 1.]
        return cfg

    # ---------------- callbacks ----------------

    def _tracks_cb(self, msg):
        self._tracks = [TrackView(t) for t in msg.tracks]
        self._tracks_stamp = msg.header.stamp

    def _color_cb(self, msg):
        self._color = self._bridge.imgmsg_to_cv2(msg, 'rgb8')
        self._color_stamp = msg.header.stamp
        self._color_frame = msg.header.frame_id

    def _depth_cb(self, msg):
        if msg.encoding not in ('16UC1', '32FC1', '64FC1'):
            self._depth = None
            self.get_logger().warning('Unsupported depth encoding: '+msg.encoding, throttle_duration_sec=2.)
            return
        img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        if msg.encoding in ('32FC1', '64FC1'):
            # Gazebo publishes metres; the network and backprojection expect mm.
            img = np.where(np.isfinite(img) & (img > 0), img*1000., 0.).astype(np.float32)
        self._depth = img
        self._depth_stamp = msg.header.stamp
        self._depth_frame = msg.header.frame_id

    def _info_cb(self, msg):
        self._K = np.array(msg.k, float).reshape(3, 3)
        self._color_info = msg

    def _depth_info_cb(self, msg):
        self._depth_info = msg

    def _feedback_cb(self, msg):
        self._feedback_time = self.get_clock().now()

    def _arm_status_cb(self, msg):
        self._arm_status = msg
        self._arm_status_time = self.get_clock().now()

    def _reset_cb(self, req, resp):
        if req.data:
            self._fsm.reset_halt()
            self._watchdog_tripped = False
            self.get_logger().info('FSM reset from HALT')
        resp.success = True
        return resp

    def _publish_state(self):
        msg = String()
        if self._static_once is not None:
            msg.data = self._static_once.state
            self._pub_state.publish(msg)
            return
        msg.data = (f'{self._fsm.state} misses={self._fsm.misses} '
                    f'successes={self._fsm.successes}')
        self._pub_state.publish(msg)
        for line in self._fsm._log[-5:]:
            self.get_logger().info(line)
        self._fsm._log.clear()

    # ---------------- main loop ----------------

    def _tick(self):
        if self._static_once is not None:
            self._static_once.tick()
            return
        if not self.get_parameter('execution_enabled').value or self._p['dry_run']:
            if self._continuous_motion_issued and not self._watchdog_tripped:
                self._stop_continuous('Execution disabled during continuous grasp')
            return
        if self._watchdog_tripped or self._fsm.state == HALT:
            return
        # arm feedback watchdog
        now = self.get_clock().now()
        if (now - self._feedback_time).nanoseconds * 1e-9 > self._p['feedback_timeout']:
            if not self._watchdog_tripped:
                self._stop_continuous('ARM FEEDBACK TIMEOUT - stop requested')
            return

        if (now-self._arm_status_time).nanoseconds*1e-9 > self._p['feedback_timeout']:
            self._stop_continuous('ARM STATUS TIMEOUT - stop requested')
            return
        if self._arm_status is None:
            return
        if self._arm_status.err_status or self._arm_status.arm_status in (1, 5, 7, 9, 10):
            self._stop_continuous('Driver fault - stop requested')
            return

        # request inference for the tracked target (rate-limited, latest-only)
        self._maybe_request_inference(now)

        grasp = None
        if self._inference is not None:
            res = self._inference.latest(max_age=self._p['result_max_age'])
            if res is not None and res.get('status') in (0, 1):
                grasp = res
                if res['status'] == 0:
                    self._publish_candidate(res)

        inp = dict(
            now=now.nanoseconds * 1e-9,
            tracks=[t for t in self._tracks
                    if now.nanoseconds * 1e-9 - t.stamp < 1.0],
            grasp=grasp,
            arm_done=self._arm.is_motion_done(),
            gripper_width=self._arm.get_gripper_width(),
            joint_state=self._arm.get_joint_state(),
        )
        try:
            cmds = self._fsm.tick(inp)
        except Exception as error:
            self._stop_continuous('Continuous grasp failed: '+str(error))
            return
        if (self._fsm.last_delta_s is not None and
                self._fsm.last_delta_s != self._last_pub_delta):
            self._last_pub_delta = self._fsm.last_delta_s
            msg = Float64()
            msg.data = self._fsm.last_delta_s
            self._pub_delta_s.publish(msg)
            self.get_logger().info(f'delta_s at grasp: '
                                   f'{self._fsm.last_delta_s * 1000:.1f} mm')
        for c in cmds:
            try:
                self._execute(c)
            except Exception as error:
                self._stop_continuous('Command rejected: '+str(error))
            if self._watchdog_tripped:
                break

    def _maybe_request_inference(self, now):
        if self._inference is None or self._K is None:
            return
        if self._color is None or self._depth is None or self._color_stamp is None:
            return
        image_stamp = self._color_stamp.sec+self._color_stamp.nanosec*1e-9
        if self._depth_stamp is None:
            return
        depth_stamp = self._depth_stamp.sec+self._depth_stamp.nanosec*1e-9
        capture_time = now.nanoseconds*1e-9
        if (abs(image_stamp-depth_stamp) > .02
                or not 0 <= capture_time-image_stamp <= .2
                or not 0 <= capture_time-depth_stamp <= .2):
            return
        if not self._registered_images_valid():
            return
        if self._fsm.state not in ('TRACK', 'APPROACH', 'WAIT'):
            return
        tr = next((t for t in self._tracks if t.id == self._fsm.target_id), None)
        if tr is None or not 0 <= capture_time-tr.stamp <= .2:
            return
        # TF camera_optical -> base at the image stamp
        try:
            tf = self._tf_buffer.lookup_transform(
                self._p['base_frame'], self._p['camera_optical_frame'],
                rclpy.time.Time.from_msg(self._color_stamp),
                timeout=rclpy.duration.Duration(seconds=0.1))
        except TransformException as e:
            self.get_logger().debug(f'TF for inference failed: {e}')
            return
        q = tf.transform.rotation
        R = _quat_R(q.w, q.x, q.y, q.z)
        t = np.array([tf.transform.translation.x, tf.transform.translation.y,
                      tf.transform.translation.z])
        self._inference.request(
            self._color, self._depth, tr.u_px, tr.v_px, self._K,
            image_stamp, {'R': R, 't': t}, belt_z=tr.belt_z, target_id=tr.id,
            distortion=self._color_info.d, distortion_model=self._color_info.distortion_model)

    def _registered_images_valid(self):
        color, depth = self._color_info, self._depth_info
        if color is None or depth is None:
            return False
        try:
            return (color.header.frame_id == depth.header.frame_id == self._color_frame
                    == self._depth_frame == self._p['camera_optical_frame']
                    and (color.width, color.height) == (depth.width, depth.height)
                    == (self._color.shape[1], self._color.shape[0])
                    == (self._depth.shape[1], self._depth.shape[0])
                    and np.allclose(color.k, depth.k, atol=1e-4, rtol=0)
                    and len(color.d) == len(depth.d)
                    and np.allclose(color.d, depth.d, atol=1e-6, rtol=0)
                    and color.distortion_model == depth.distortion_model
                    and color.binning_x in (0, 1) and color.binning_y in (0, 1)
                    and depth.binning_x in (0, 1) and depth.binning_y in (0, 1)
                    and not any((color.roi.x_offset, color.roi.y_offset, color.roi.width, color.roi.height,
                                 depth.roi.x_offset, depth.roi.y_offset, depth.roi.width, depth.roi.height))
                    and np.isfinite(self._K).all() and self._K[0, 0] > 0 and self._K[1, 1] > 0)
        except (TypeError, ValueError):
            return False

    def _publish_candidate(self, res):
        msg = GraspCandidate()
        c = res['center']
        msg.center.x, msg.center.y, msg.center.z = float(c[0]), float(c[1]), float(c[2])
        msg.yaw = res['yaw']
        msg.width_mm = res['width_mm']
        msg.score = res.get('score', 0.0)
        msg.u_px = res.get('u_px', 0.0)
        msg.v_px = res.get('v_px', 0.0)
        msg.status = res['status']
        self._pub_candidate.publish(msg)

    # ---------------- command execution + safety ----------------

    def _execute(self, cmd):
        kind = cmd['type']
        if kind == 'log':
            return
        if kind == 'estop':
            self._stop_continuous('Continuous grasp halted')
            return
        if self._watchdog_tripped:
            self.get_logger().warn(f'blocked {kind}: watchdog tripped')
            return
        self._continuous_motion_issued = True
        if kind in ('move_p', 'move_l'):
            pose = cmd['pose']
            if not self._pose_ok(pose, kind):
                self.get_logger().error(f'rejected unsafe {kind}: {pose}')
                self._stop_continuous('Pose outside configured workspace or step limit')
                return
            self._last_cmd_pose = pose
            self._arm.move_p(pose) if kind == 'move_p' else self._arm.move_l(pose)
        elif kind == 'move_j':
            self._last_cmd_pose = None
            self._arm.move_j(cmd['joints'])
        elif kind == 'gripper':
            self._arm.move_gripper(cmd['width'], cmd.get('force', 1.0))
        elif kind == 'home':
            self._last_cmd_pose = None
            self._arm.home()

    def _pose_ok(self, pose, kind):
        ws = self._p['workspace']
        x, y, z = pose[:3]
        if not (ws['x_min'] <= x <= ws['x_max'] and
                ws['y_min'] <= y <= ws['y_max'] and
                z >= ws['z_min']):
            return False
        if self._last_cmd_pose is not None:
            d = math.dist(pose[:3], self._last_cmd_pose[:3])
            if d > self._p['max_target_step']:
                return False
        return True


_DEFAULTS = {
    'infer_target_radius_px': 60.0,
    'backend': 'ros_topic', 'topic_prefix': '',
    'tracks_topic': '/perception/tracks',
    'color_topic': '/camera/color/image_raw',
    'depth_topic': '/camera/depth/image_raw',  # aligned when depth_registration:=true
    'color_camera_info': '/camera/color/camera_info',
    'depth_camera_info': '/camera/depth/camera_info',
    'calibration_file': '',
    'qos_reliability': 'best_effort',
    'visualization_enabled': True,
    'visualization_topic': '/perception/debug_image',
    'base_frame': 'base_link',
    'camera_optical_frame': 'camera_color_optical_frame',
    'model_path': '/home/ubuntu22/grasp/robotic-grasping/trained-models/jacquard-rgbd-grconvnet3-drop0-ch32',
    'infer_max_rate': 10.0, 'result_max_age': 0.2, 'q_threshold': 0.25,
    'inference_device': 'cuda',
    'belt_file': '',
    'approach_height': 0.08, 'descend_offset': 0.04, 'lift_height': 0.12,
    't_approach': 1.5, 'tau_desc': 0.5, 'k_lead': 0.05,
    'gate_tol': 0.02, 'abort_tol': 0.025,
    'yaw_off': 0.0, 'grasp_roll': math.pi, 'grasp_pitch': 0.0,
    'fallback_width_mm': 40.0, 'gripper_force': 1.0,
    'speed_min': 0.01, 'speed_max': 0.10,
    'timeout_approach': 3.0, 'timeout_wait': 5.0, 'timeout_descend': 2.0,
    'timeout_grasp': 1.5, 'timeout_lift': 3.0, 'timeout_place': 10.0,
    'miss_halt_threshold': 3, 'dry_run': False, 'fsm_rate': 50.0,
    'bin_pre_joints': [], 'bin_drop_pose': [],
    'workspace': {'x_min': -0.45, 'x_max': 0.45, 'y_min': -0.35,
                  'y_max': 0.35, 'z_min': -0.05},
    'max_target_step': 0.35, 'feedback_timeout': 0.2,
    'arm_speed_percent': 50,
}


def _quat_R(w, x, y, z):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def main(args=None):
    # Preserve the ROS context until the controller can send its stop request.
    def stop_requested(signum, frame):
        raise KeyboardInterrupt

    previous_sigterm = signal.signal(signal.SIGTERM, stop_requested)
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = None
    try:
        node = GraspControllerNode()
        if node.get_parameter('exit_on_complete').value is True:
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=.1)
                actor = node._static_once
                if actor is not None and actor.state.endswith(('_DONE', '_ABORT')):
                    node._publish_state()
                    break
        else:
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if node is not None:
                node._timer.cancel()
                if node._static_once is not None:
                    node._static_once.shutdown()
                elif node._continuous_motion_issued:
                    future = node._stop_continuous('Controller shutting down')
                    if future is not None and node.context.ok():
                        rclpy.spin_until_future_complete(node, future, timeout_sec=.5)
                if node._inference is not None:
                    node._inference.shutdown()
                node.destroy_node()
        finally:
            rclpy.try_shutdown()
            signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == '__main__':
    main()
