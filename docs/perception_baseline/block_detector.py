"""Block detection node: belt-plane segmentation + tracking + speed estimation.

Pipeline per color frame (depth aligned to color grid):
1. RANSAC-fit the belt plane in depth space (refresh every N frames)
2. Mask pixels whose depth is above the plane by [h_min, h_max] -> blocks
3. Connected components -> blobs (centroid px, median height, area)
4. Backproject blob centroids to 3D (camera optical frame) -> TF to base_link
5. Multi-block tracker + Kalman XY motion estimate and covariance
6. Publish BlockTrackArray + debug image
"""
import math
import yaml
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import tf2_ros
from tf2_ros import TransformException, ConnectivityException, ExtrapolationException
from nero7_msgs.msg import BlockTrack, BlockTrackArray

from .alignment import make_depth_source
from .conveyor_model import BeltPlane, BeltModel
from .block_tracker import BlockTracker
from .speed_overlay import draw_speed_overlay


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
            ('block_height_min', 0.008),
            ('block_height_max', 0.060),
            ('min_area_px', 300),
            ('tracker_max_vel', 0.15),
            ('tracker_gate_radius', 0.03),
            ('tracker_confirm_frames', 2),
            ('tracker_lost_frames', 6),
            ('tracker_lost_timeout', 0.5),
            ('speed_window_s', 0.6),
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
            ('belt_file', '/home/ubuntu22/Nero7_grasping/src/nero7_bringup/config/belt.yaml'),
            ('calibration_file', '/home/ubuntu22/Nero7_grasping/src/nero7_bringup/config/calibration.yaml'),
        ]:
            params[name] = self.declare_parameter(name, file_params.get(name, default)).value

        self._p = params
        self._bridge = CvBridge()
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        qos = QoSProfile(depth=5)
        if params['qos_reliability'] == 'best_effort':
            qos.reliability = ReliabilityPolicy.BEST_EFFORT

        self._color = None
        self._color_stamp = None
        self.create_subscription(Image, params['color_topic'], self._color_cb, qos)
        self._info_sub = self.create_subscription(CameraInfo, params['color_camera_info'], self._info_cb, qos)
        self._K = None

        self._depth_source = make_depth_source(self, params)

        # Belt model setup
        belt = {}
        if params['belt_file']:
            try:
                with open(params['belt_file']) as f:
                    belt = yaml.safe_load(f) or {}
            except OSError as e:
                self.get_logger().warn(f'cannot read belt file: {e}')
        self._belt = BeltModel.from_yaml(belt)

        # Depth-to-Color (d2c) pixel offset registration
        self._d2c = (0, 0)
        try:
            with open(params.get('calibration_file', '')) as f:
                data = yaml.safe_load(f) or {}
            if data.get('d2c_offset'):
                self._d2c = (int(data['d2c_offset'][0]), int(data['d2c_offset'][1]))
                self.get_logger().info(f'd2c_offset: {self._d2c}')
        except OSError:
            pass

        # Workspace limits filter in base frame
        self._ws = dict(x=params.get('workspace_x', [-0.7, 0.7]),
                        y=params.get('workspace_y', [-0.7, 0.7]),
                        z=params.get('workspace_z', [-0.25, 0.15]))

        self._plane = BeltPlane(distance=params['plane_ransac_distance'],
                                min_ratio=params['plane_ransac_min_ratio'])
        self._frame_count = 0
        roi = params['roi_px']
        self._roi = tuple(roi) if any(roi) else None

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

        self._status_timer = self.create_timer(2.0, self._status_cb)
        self.get_logger().info('block_detector started')

    # ---------------- callbacks ----------------

    def _color_cb(self, msg):
        self._color = self._bridge.imgmsg_to_cv2(msg, 'bgr8')
        self._color_stamp = msg.header.stamp
        self._process()

    def _info_cb(self, msg):
        self._K = np.array(msg.k, float).reshape(3, 3)

    def _status_cb(self):
        n = len(self._tracker.tracks)
        self.get_logger().info(f'tracks: {n}', throttle_duration_sec=4.0)

    # ---------------- pipeline ----------------

    def _process(self):
        if self._color is None or self._K is None:
            return
        depth = self._depth_source.latest()
        if depth is None:
            return
        depth_img, stamp = depth

        # Time synchronization check
        if self._color_stamp is not None:
            dt = abs(self._stamp_sec(stamp) - self._stamp_sec(self._color_stamp))
            if dt > 0.1:
                return

        if depth_img.shape[:2] != self._color.shape[:2]:
            depth_img = cv2.resize(depth_img,
                                   (self._color.shape[1], self._color.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)

        if self._d2c != (0, 0):
            depth_img = shift_depth(depth_img, *self._d2c)

        # Batch TF lookup once per frame for better efficiency
        tf_mat = self._get_camera_to_base_tf(stamp)
        if tf_mat is None:
            return
        R_cam2base, t_cam2base = tf_mat

        # Plane fitting refresh
        refit = (self._frame_count % self._p['plane_refresh_every'] == 0
                 or self._plane.coeffs is None)
        if refit:
            self._plane.fit(depth_img, roi=self._roi)
        self._frame_count += 1

        if self._plane.coeffs is None:
            return

        # Plane z evaluation across image space
        h, w = depth_img.shape
        u_grid, v_grid = np.meshgrid(np.arange(w), np.arange(h))
        plane_z = (self._plane.coeffs[0] * u_grid +
                   self._plane.coeffs[1] * v_grid +
                   self._plane.coeffs[2])

        height = plane_z - depth_img  # Height > 0 is above the conveyor belt
        mask = ((height >= self._p['block_height_min']) &
                (height <= self._p['block_height_max'])).astype(np.uint8)
        if self._roi is not None:
            u0, v0, u1, v1 = self._roi
            roi_mask = np.zeros_like(mask)
            roi_mask[v0:v1, u0:u1] = 1
            mask &= roi_mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        detections = []
        
        fx, fy = self._K[0, 0], self._K[1, 1]
        cx, cy = self._K[0, 2], self._K[1, 2]

        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self._p['min_area_px']:
                continue
            u, v = centroids[i]
            width_px = math.sqrt(area)
            ys, xs = np.nonzero(labels == i)
            
            h_median = float(np.median(height[ys, xs]))
            z_top = float(np.median(depth_img[ys, xs]))
            if not np.isfinite(z_top) or z_top <= 0:
                continue

            # Backproject 2D centroid to 3D Camera Frame
            p_cam = np.array([(u - cx) * z_top / fx,
                              (v - cy) * z_top / fy,
                              z_top])
            
            # Transform to Base Link Frame
            p_base = R_cam2base @ p_cam + t_cam2base

            # Workspace range filtering
            if not (self._ws['x'][0] <= p_base[0] <= self._ws['x'][1] and
                    self._ws['y'][0] <= p_base[1] <= self._ws['y'][1] and
                    self._ws['z'][0] <= p_base[2] <= self._ws['z'][1]):
                continue

            # Belt surface height under the block in base_link (for approach z calculation)
            z_plane = plane_z[int(round(v)), int(round(u))]
            p_plane = np.array([(u - cx) * z_plane / fx,
                                (v - cy) * z_plane / fy,
                                z_plane])
            p_belt = R_cam2base @ p_plane + t_cam2base
            belt_z = float(p_belt[2])

            detections.append(dict(u=float(u), v=float(v),
                                   centroid_base=p_base,
                                   height=h_median,
                                   width_px=width_px,
                                   area_px=float(area),
                                   belt_z=belt_z))

        tracks = self._tracker.update(detections, self._stamp_sec(stamp))
        self._publish_tracks(tracks, stamp)

        if self._pub_debug is not None:
            self._publish_debug(tracks, stamp)

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

    def _publish_debug(self, tracks, stamp):
        img = draw_speed_overlay(self._color, tracks)
        msg = self._bridge.cv2_to_imgmsg(img, 'bgr8')
        msg.header.stamp = stamp
        msg.header.frame_id = self._p['camera_optical_frame']
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
