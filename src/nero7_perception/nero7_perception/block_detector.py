"""Synchronized registered RGB-D detection with switchable base belt geometry."""
from collections import deque
import json
import time
import yaml
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from cv_bridge import CvBridge
import tf2_ros
from tf2_ros import TransformException, ConnectivityException, ExtrapolationException
from nero7_msgs.msg import BlockTrack, BlockTrackArray

from .alignment import make_depth_source
from .geometry import CameraGeometry, check_transform
from .base_plane import (EmptyBeltCalibration, load_plane, save_plane,
                         plane_context, check_plane_context, archive_plane)
from .detection import (DETECTION_DEFAULTS, validate_detection_config, resolve_detection_paths, calibration_points,
                        detect_blocks, detect_legacy)
from .conveyor_model import BeltPlane, BeltModel
from .block_tracker import BlockTracker
from .speed_overlay import draw_speed_overlay
from .frame_selection import newest_pair


class BlockDetectorNode(Node):
    def __init__(self):
        super().__init__('block_detector')
        p = self.declare_parameter('params_file', '').value
        file_params = {}
        if p:
            with open(p) as f:
                file_params = yaml.safe_load(f)['block_detector']['ros__parameters']
        params = {}
        for name, default in [
            ('color_topic', '/camera/color/image_raw'),
            ('depth_topic', '/camera/aligned_depth_to_color/image_raw'),
            ('points_topic', '/camera/depth/points'),
            ('color_camera_info', '/camera/color/camera_info'),
            ('depth_camera_info', '/camera/depth/camera_info'),
            ('qos_reliability', 'best_effort'),
            ('alignment_mode', 'aligned_topic'),
            ('roi_px', [0, 0, 0, 0]),
            ('plane_ransac_distance', 0.006),
            ('plane_ransac_min_ratio', 0.4),
            ('plane_refresh_every', 10),
            ('plane_ransac_max_points', 5000),
            ('processing_rate_hz', 30.0),
            ('debug_rate_hz', 10.0),
            ('debug_max_width', 0),
            ('tracker_reacquire_samples', 3),
            ('tracker_reacquire_max_distance', .02),
            ('tracker_reacquire_consistency', .006),
            ('block_height_min', 0.008),
            ('block_height_max', 0.060),
            ('min_area_px', 300),
            ('tracker_max_vel', 0.15),
            ('tracker_gate_radius', 0.03),
            ('tracker_confirm_frames', 2),
            ('tracker_lost_frames', 6),
            ('tracker_lost_timeout', 0.5),
            ('speed_window_s', 1.2),
            ('speed_min_samples', 8),
            ('speed_min_span', 0.3),
            ('speed_resid_std_max', 0.008),
            ('speed_jump_clear', 0.1),
            ('kalman_measurement_std', 0.002),
            ('kalman_accel_noise_std', 0.002),
            ('kalman_cross_accel_noise_std', 0.0005),
            ('kalman_initial_velocity_std', 0.10),
            ('kalman_innovation_gate', 9.210340371976184),
            ('kalman_max_velocity_std', 0.010),
            ('kalman_observation_max_age', 0.15),
            ('kalman_qualification_gap', 0.3),
            ('workspace_x', [-0.7, 0.7]),
            ('workspace_y', [-0.7, 0.7]),
            ('workspace_z', [-0.25, 0.15]),
            ('base_frame', 'base_link'),
            ('camera_optical_frame', 'camera_color_optical_frame'),
            ('publish_debug', True),
            ('debug_topic', '/perception/debug_image'),
            ('belt_file', ''),
            ('calibration_file', ''),
        ] + list(DETECTION_DEFAULTS.items()):
            value = file_params.get(name, default)
            if name in ('belt_file', 'calibration_file') and not value:
                value = resolve_detection_paths(file_params, p or None)[name]
            params[name] = self.declare_parameter(name, value).value

        params = resolve_detection_paths(params, p or None)
        validate_detection_config(params)
        if (not np.isfinite([params['processing_rate_hz'], params['debug_rate_hz']]).all()
                or params['processing_rate_hz'] <= 0 or params['debug_rate_hz'] <= 0
                or params['debug_max_width'] < 0):
            raise ValueError('Invalid processing/debug rate or debug width')
        self._p = params
        self._bridge = CvBridge()
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        qos = QoSProfile(depth=1)
        if params['qos_reliability'] == 'best_effort':
            qos.reliability = ReliabilityPolicy.BEST_EFFORT

        self._color = None
        self._color_stamp = None
        self._colors = deque(maxlen=params['sync_buffer_size'])
        self._camera = None
        self._last_depth_stamp = -np.inf
        self._last_color_stamp = -np.inf
        self._processing = False
        self._diagnostics = dict(processed_frames=0, rejected_frames=0, backlog_frames_dropped=0,
                                 color_frames_received=0, debug_frames_published=0)
        self._last_debug_time = -np.inf
        self.create_subscription(Image, params['color_topic'], self._color_cb, qos)
        self._info_sub = self.create_subscription(CameraInfo, params['color_camera_info'], self._info_cb, qos)
        self._K = None

        self._depth_source = make_depth_source(self, params)

        # Belt model setup
        belt = {}
        if params['belt_file']:
            with open(params['belt_file']) as f:
                belt = yaml.safe_load(f) or {}
        self._belt = BeltModel.from_yaml(belt)

        # Depth-to-Color (d2c) pixel offset registration
        self._d2c = (0, 0)
        try:
            with open(params.get('calibration_file', '')) as f:
                data = yaml.safe_load(f) or {}
            if params['apply_d2c_offset'] and data.get('d2c_offset'):
                self._d2c = (int(data['d2c_offset'][0]), int(data['d2c_offset'][1]))
                self.get_logger().info(f'd2c_offset: {self._d2c}')
                self.get_logger().warning('Pixel offset is an empirical correction, not geometric registration')
        except OSError:
            pass

        # Workspace limits filter in base frame
        self._ws = dict(x=params.get('workspace_x', [-0.7, 0.7]),
                        y=params.get('workspace_y', [-0.7, 0.7]),
                        z=params.get('workspace_z', [-0.25, 0.15]))

        self._plane = BeltPlane(distance=params['plane_ransac_distance'],
                                min_ratio=params['plane_ransac_min_ratio'], max_points=params['plane_ransac_max_points'])
        self._frame_count = 0
        roi = params['roi_px']
        self._roi = tuple(roi) if any(roi) else None
        self._base_plane, self._plane_context = None, None
        self._calibration_context = None
        self._calibration_done = False
        self._calibrator = EmptyBeltCalibration(
            mode=params['plane_mode'], frames=params['plane_calibration_frames'],
            distance=params['plane_ransac_distance'], min_ratio=params['base_plane_min_ratio'],
            max_tilt=params['plane_max_tilt_rad'], max_flat_residual=params['plane_max_flat_residual_m'],
            max_height_spread=params['plane_max_height_spread_m'], min_xy_span=params['plane_min_xy_span_m'],
            max_points=params['plane_calibration_max_points'], max_3d_tilt=params['plane_max_3d_tilt_rad'])
        if params['plane_calibration_enabled']:
            if not params['plane_file'] or not any(params['plane_roi_px']):
                raise ValueError('Calibration requires plane_file and an explicit empty-belt plane_roi_px')
            self._diagnostics['previous_plane_file'] = archive_plane(params['plane_file'])
        elif params['detection_mode'] != 'legacy_depth':
            if params['plane_file']:
                self._base_plane, self._plane_context = load_plane(params['plane_file'])
                if self._base_plane.mode != params['plane_mode']:
                    raise ValueError('plane_mode does not match the calibrated file')
            else:
                self.get_logger().warning('New detection mode has no plane_file; no tracks will be published')
        else:
            self.get_logger().warning('legacy_depth baseline retains optical-axis height; validate before motion')

        speed_cfg = dict(
            window_s=params['speed_window_s'],
            min_samples=params['speed_min_samples'],
            min_span=params['speed_min_span'],
            resid_std_max=params['speed_resid_std_max'],
            jump_clear=params['speed_jump_clear'],
        )
        self._tracker = BlockTracker(
            self._belt, speed_cfg,
            max_vel=params['tracker_max_vel'],
            gate_radius=params['tracker_gate_radius'],
            confirm_frames=params['tracker_confirm_frames'],
            lost_frames=params.get('tracker_lost_frames', 6),
            lost_timeout=params.get('tracker_lost_timeout', 0.5),
            reacquire_samples=params['tracker_reacquire_samples'],
            reacquire_max_distance=params['tracker_reacquire_max_distance'],
            reacquire_consistency=params['tracker_reacquire_consistency'],
            motion_cfg=dict(
                measurement_std=params.get('kalman_measurement_std', 0.002),
                accel_noise_std=params.get('kalman_accel_noise_std', 0.002),
                cross_accel_noise_std=params.get('kalman_cross_accel_noise_std', 0.0005),
                initial_velocity_std=params.get('kalman_initial_velocity_std', 0.10),
                innovation_gate=params.get('kalman_innovation_gate', 9.210340371976184),
                max_velocity_std=params.get('kalman_max_velocity_std', 0.010),
                observation_max_age=params.get('kalman_observation_max_age', 0.15),
                qualification_gap=params.get('kalman_qualification_gap', 0.3)))

        self._pub_tracks = self.create_publisher(BlockTrackArray, '/perception/tracks', 5)
        self._pub_debug = None
        if params['publish_debug']:
            self._pub_debug = self.create_publisher(Image, params['debug_topic'], 5)

        self._processing_timer = self.create_timer(1./params['processing_rate_hz'], self._process)
        self._status_timer = self.create_timer(2.0, self._status_cb)
        self._pub_diagnostics = self.create_publisher(String, '/perception/diagnostics', 5)
        self._started_monotonic = time.perf_counter()
        self._status_last_time, self._status_last_processed = self._started_monotonic, 0
        self.get_logger().info('block_detector started')

    # ---------------- callbacks ----------------

    def _color_cb(self, msg):
        stamp = self._stamp_sec(msg.header.stamp)
        if stamp <= self._last_color_stamp:
            return
        image = self._bridge.imgmsg_to_cv2(msg, 'bgr8')
        self._colors.append((image, msg.header.stamp, msg.header.frame_id))
        self._diagnostics['color_frames_received'] += 1

    def _info_cb(self, msg):
        try:
            camera = CameraGeometry.from_info(msg, self._p['color_rectified'])
            if camera.frame_id != self._p['camera_optical_frame']:
                raise ValueError('CameraInfo does not belong to configured color optical frame')
            old = self._camera
            if (old is None or not np.array_equal(old.K, camera.K) or old.D != camera.D
                    or old.distortion_model != camera.distortion_model
                    or (old.width, old.height, old.frame_id) != (camera.width, camera.height, camera.frame_id)):
                self._camera = camera
            self._K = camera.K
        except ValueError as error:
            self._camera = None
            self.get_logger().warning(str(error), throttle_duration_sec=2.)

    def _status_cb(self):
        now = time.perf_counter()
        processed = self._diagnostics['processed_frames']
        report = dict(self._diagnostics, tracks=len(self._tracker.tracks),
                      detection_mode=self._p['detection_mode'], plane_mode=self._p['plane_mode'],
                      height_semantics=('optical_depth_difference' if self._p['detection_mode'] == 'legacy_depth'
                                        else 'base_vertical_m'),
                      processed_fps=processed / max(now-self._started_monotonic, 1e-6),
                      processed_fps_window=(processed-self._status_last_processed) /
                          max(now-self._status_last_time, 1e-6))
        self._status_last_time, self._status_last_processed = now, processed
        if self._base_plane is not None:
            report.update(plane_normal_base=self._base_plane.normal.tolist(), plane_offset_m=self._base_plane.offset)
        report['track_created_total'] = self._tracker._next_id
        report['speed_window_s'] = self._p['speed_window_s']
        report['tracks_detail'] = [
            dict(id=tr.id, status=tr.status, misses=tr.misses,
                 observation_age_s=max(0., self.get_clock().now().nanoseconds*1e-9-tr.stamp),
                 accepted_recent=len(tr.motion.accepted_stamps), rejected_total=tr.motion.rejected_count,
                 velocity_std_m_s=float(np.sqrt(np.linalg.eigvalsh(tr.motion_covariance[2:, 2:])[-1])),
                 speed_valid=bool(tr.speed_valid),
                 qualification_reason=tr.motion.qualification_reason,
                 reacquisitions=tr.reacquisitions)
            for tr in self._tracker.tracks]
        msg = String()
        msg.data = json.dumps(report, allow_nan=False)
        self._pub_diagnostics.publish(msg)
        self.get_logger().info(msg.data, throttle_duration_sec=4.)

    def _process(self):
        if self._camera is None or self._processing:
            return
        self._processing = True
        try:
            now = self.get_clock().now().nanoseconds*1e-9
            pair = newest_pair(self._colors, self._depth_source, self._p['sync_max_delta_s'],
                               self._last_color_stamp, self._last_depth_stamp, now,
                               self._p['image_max_age_s'])
            if pair is None:
                return
            color, color_stamp, color_frame, frame = pair
            color_time = self._stamp_sec(color_stamp)
            older = sum(self._last_color_stamp < self._stamp_sec(entry[1]) < color_time
                        for entry in self._colors)
            self._diagnostics['backlog_frames_dropped'] = self._diagnostics.get('backlog_frames_dropped', 0)+older
            depth_time = self._stamp_sec(frame.stamp)
            self._last_color_stamp, self._last_depth_stamp = color_time, depth_time
            now = self.get_clock().now().nanoseconds * 1e-9
            started = time.perf_counter()
            try:
                if not (-self._p['sync_max_delta_s'] <= now-depth_time <= self._p['image_max_age_s']):
                    raise ValueError('RGB-D capture timestamp is stale or in the future')
                self._camera.check_grid(color, color_frame)
                self._depth_source.check_camera(self._camera, frame)
                depth = frame.depth
                if self._d2c != (0, 0):
                    depth = shift_depth(depth, *self._d2c)
                tf_mat = self._get_camera_to_base_tf(frame.stamp)
                if tf_mat is None:
                    raise ValueError('No optical-to-base TF at depth capture time')
                transform = np.eye(4)
                transform[:3, :3], transform[:3, 3] = tf_mat
                check_transform(transform)
                self._color, self._color_stamp = color, color_stamp
                if self._p['plane_calibration_enabled']:
                    self._calibrate(color, depth, transform, depth_time)
                    return  # Calibration never publishes candidate tracks.
                if self._p['detection_mode'] == 'legacy_depth':
                    detections, _ = detect_legacy(color, depth, self._camera, transform,
                        self._plane, self._p,
                        refit=self._frame_count % self._p['plane_refresh_every'] == 0)
                    if self._plane.coeffs is None:
                        raise ValueError('Legacy plane fit unavailable')
                else:
                    if self._plane_context is None:
                        raise ValueError('No validated empty-belt plane file')
                    check_plane_context(self._plane_context, self._camera, transform,
                                        self._p['base_frame'], self._d2c)
                    detections, _ = detect_blocks(color, depth, self._camera, transform,
                                                  self._base_plane, self._p)
                self._frame_count += 1
                tracks = self._tracker.update(detections, depth_time)
                self._publish_tracks(tracks, frame.stamp)
                self._publish_debug_if_due(tracks, frame.stamp)
                self._diagnostics.update(processed_frames=self._diagnostics['processed_frames'] + 1,
                    detections=len(detections), process_ms=(time.perf_counter()-started)*1000,
                    sync_delta_ms=abs(color_time-depth_time)*1000,
                    capture_age_ms=(self.get_clock().now().nanoseconds*1e-9-depth_time)*1000,
                    last_error='')
            except (ValueError, OSError, cv2.error) as error:
                self._diagnostics['rejected_frames'] += 1
                self._diagnostics['last_error'] = str(error)
                self.get_logger().warning(f'Detection rejected: {error}', throttle_duration_sec=2.)
                self._publish_debug_if_due([], color_stamp, image=color,
                                           frame_id=color_frame, error=str(error))
        finally:
            self._processing = False

    def _calibrate(self, color, depth, transform, stamp):
        if self._calibration_done:
            return
        if self._calibration_context is None:
            self._calibration_context = plane_context(self._camera, transform, self._p['base_frame'], self._d2c)
        check_plane_context(self._calibration_context, self._camera, transform, self._p['base_frame'], self._d2c)
        points = calibration_points(color, depth, self._camera, transform, self._p)
        ready = self._calibrator.add(points, stamp)
        self._diagnostics['calibration_frames'] = len(self._calibrator.samples)
        if ready:
            # A failed fit stops accumulation; restart after correcting ROI/TF/mode.
            self._calibration_done = True
            plane = self._calibrator.fit()
            save_plane(self._p['plane_file'], plane, self._camera, transform,
                       self._p['base_frame'], self._p['plane_roi_px'], self._d2c)
            self._diagnostics['plane'] = plane.diagnostics
            self._diagnostics['calibration_saved'] = self._p['plane_file']
            self.get_logger().info('Empty belt calibration saved; stop this node and restart detection')

    def _get_camera_to_base_tf(self, stamp):
        """Lookup transform matrix from camera optical frame to base_link frame."""
        try:
            tf = self._tf_buffer.lookup_transform(
                self._p['base_frame'], self._p['camera_optical_frame'],
                rclpy.time.Time.from_msg(stamp),
                timeout=rclpy.duration.Duration(seconds=0.05))
        except (TransformException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().debug(f'TF lookup failed: {e}')
            return None

        q = tf.transform.rotation
        R = _quat_R(q.w, q.x, q.y, q.z)
        t = np.array([tf.transform.translation.x, 
                      tf.transform.translation.y, 
                      tf.transform.translation.z])
        return R, t

    @staticmethod
    def _stamp_sec(stamp):
        return stamp.sec + stamp.nanosec * 1e-9

    # ---------------- output ----------------

    def _publish_tracks(self, tracks, stamp):
        msg = BlockTrackArray()
        msg.header.stamp = stamp
        msg.header.frame_id = self._p['base_frame']
        for tr in tracks:
            t = BlockTrack()
            t.id = tr.id
            # Lost tracks must retain the timestamp of the last observation.
            stamp_ns = int(round(tr.stamp * 1e9))
            t.stamp.sec, t.stamp.nanosec = divmod(stamp_ns, 1000000000)
            motion_ns = int(round(tr.motion_stamp * 1e9))
            t.motion_stamp.sec, t.motion_stamp.nanosec = divmod(motion_ns, 1000000000)
            t.centroid_base.x = tr.centroid[0]
            t.centroid_base.y = tr.centroid[1]
            t.centroid_base.z = tr.centroid[2]
            t.observed_centroid_base.x = tr.observed_centroid[0]
            t.observed_centroid_base.y = tr.observed_centroid[1]
            t.observed_centroid_base.z = tr.observed_centroid[2]
            
            t.velocity_base.x, t.velocity_base.y, t.velocity_base.z = tr.velocity
            t.speed_valid = bool(tr.speed_valid)
            t.motion_covariance = tr.motion_covariance.reshape(-1).tolist()
            t.motion_process_noise = tr.motion_process_noise.reshape(-1).tolist()
            t.speed = tr.speed if (tr.speed_valid and tr.speed is not None) else 0.0
            t.s = self._belt.s_of(tr.centroid)
            t.height = tr.height
            t.belt_z = getattr(tr, 'belt_z', 0.0)
            t.width_px = tr.width_px
            t.u_px, t.v_px = tr.u_px, tr.v_px
            t.confidence = tr.confidence
            t.status = tr.status
            msg.tracks.append(t)
        self._pub_tracks.publish(msg)

    def _publish_debug_if_due(self, tracks, stamp, **kwargs):
        wall_now = time.perf_counter()
        if self._pub_debug is not None and wall_now-getattr(self, '_last_debug_time', -np.inf) >= 1./self._p['debug_rate_hz']:
            self._publish_debug(tracks, stamp, **kwargs)
            self._last_debug_time = wall_now
            self._diagnostics['debug_frames_published'] = self._diagnostics.get('debug_frames_published', 0)+1

    def _publish_debug(self, tracks, stamp, image=None, frame_id=None, error=None):
        image = self._color if image is None else image
        width = self._p['debug_max_width']
        scale = min(1., width/image.shape[1]) if width else 1.
        if scale < 1.:
            image = cv2.resize(image, (int(round(image.shape[1]*scale)),
                                      int(round(image.shape[0]*scale))), interpolation=cv2.INTER_AREA)
        img = draw_speed_overlay(image, tracks, pixel_scale=scale, error=error)
        msg = self._bridge.cv2_to_imgmsg(img, 'bgr8')
        msg.header.stamp = stamp
        msg.header.frame_id = self._p['camera_optical_frame'] if frame_id is None else frame_id
        self._pub_debug.publish(msg)


def _quat_R(w, x, y, z):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def shift_depth(dep, du, dv):
    """Shift the depth image by (du, dv) pixels; edges become NaN."""
    out = np.full_like(dep, np.nan, dtype=np.float32)
    h, w = dep.shape
    sv0, sv1 = max(0, dv), min(h, h + dv)
    su0, su1 = max(0, du), min(w, w + du)
    dv0, dv1 = max(0, -dv), min(h, h - dv)
    du0, du1 = max(0, -du), min(w, w - du)
    out[dv0:dv1, du0:du1] = dep[sv0:sv1, su0:su1]
    return out


def main(args=None):
    rclpy.init(args=args)
    node = BlockDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
