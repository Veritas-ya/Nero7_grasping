"""Capture synchronized registered RGB-D/TF to NPZ. No arm interfaces exist here."""
from collections import deque
from pathlib import Path
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import tf2_ros
from scipy.spatial.transform import Rotation

from ..alignment import make_depth_source, stamp_seconds
from ..detection import DETECTION_DEFAULTS
from ..geometry import CameraGeometry, check_transform


class CaptureRgbd(Node):
    def __init__(self):
        super().__init__('capture_rgbd')
        defaults = dict(DETECTION_DEFAULTS, color_topic='/camera/color/image_raw',
                        depth_topic='/camera/depth/image_raw', color_camera_info='/camera/color/camera_info',
                        depth_camera_info='/camera/depth/camera_info', points_topic='/camera/depth/points',
                        alignment_mode='aligned_topic', qos_reliability='best_effort',
                        camera_optical_frame='camera_color_optical_frame', base_frame='base_link')
        self.params = {k: self.declare_parameter(k, v).value for k, v in defaults.items()}
        self.limit = self.declare_parameter('capture_frames', 60).value
        self.empty = self.declare_parameter('capture_empty', False).value
        self.output = self.declare_parameter('output_file', '/tmp/nero7_rgbd.npz').value
        self.timeout = self.declare_parameter('capture_timeout_s', 30.).value
        if self.limit < 3 or self.timeout <= 0:
            raise ValueError('Need >=3 capture frames and a positive timeout')
        self.bridge, self.camera, self.frames = CvBridge(), None, []
        self.colors = deque(maxlen=self.params['sync_buffer_size'])
        self.last_depth = self.last_color = -np.inf
        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer, self)
        self.source = make_depth_source(self, self.params, self.process)
        qos = QoSProfile(depth=8)
        if self.params['qos_reliability'] == 'best_effort':
            qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.create_subscription(Image, self.params['color_topic'], self.color_cb, qos)
        self.create_subscription(CameraInfo, self.params['color_camera_info'], self.info_cb, qos)

    def info_cb(self, msg):
        try:
            camera = CameraGeometry.from_info(msg, self.params['color_rectified'])
            if camera.frame_id != self.params['camera_optical_frame']:
                raise ValueError('Color optical frame mismatch')
            self.camera = camera
        except ValueError as error:
            self.camera = None
            self.get_logger().warning(str(error), throttle_duration_sec=2.)

    def color_cb(self, msg):
        self.colors.append((self.bridge.imgmsg_to_cv2(msg, 'bgr8'), msg.header))
        self.process()

    def process(self):
        if self.camera is None or len(self.frames) >= self.limit:
            return
        for color, header in list(self.colors):
            color_time = stamp_seconds(header.stamp)
            if color_time <= self.last_color:
                continue
            depth = self.source.closest(header.stamp, self.params['sync_max_delta_s'], self.last_depth)
            if depth is None:
                continue
            try:
                self.camera.check_grid(color, header.frame_id)
                self.source.check_camera(self.camera, depth)
                depth_time = stamp_seconds(depth.stamp)
                age = self.get_clock().now().nanoseconds * 1e-9 - depth_time
                if not -self.params['sync_max_delta_s'] <= age <= self.params['image_max_age_s']:
                    raise ValueError('Capture timestamps are stale/future')
                tf = self.buffer.lookup_transform(self.params['base_frame'], self.camera.frame_id,
                    rclpy.time.Time.from_msg(depth.stamp), timeout=rclpy.duration.Duration(seconds=.05))
                transform = np.eye(4)
                q = tf.transform.rotation
                transform[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
                transform[:3, 3] = [tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z]
                check_transform(transform)
                if self.frames:
                    previous = self.frames[-1]['camera']
                    if (not np.array_equal(previous.K, self.camera.K) or previous.D != self.camera.D
                            or previous.distortion_model != self.camera.distortion_model
                            or (previous.width, previous.height) != (self.camera.width, self.camera.height)):
                        raise ValueError('Camera calibration changed during capture')
                self.frames.append(dict(color=color.copy(), depth=depth.depth.copy(),
                    stamp=depth_time, color_stamp=color_time, transform=transform, camera=self.camera))
                self.last_depth, self.last_color = depth_time, color_time
                if len(self.frames) >= self.limit:
                    break
            except (ValueError, tf2_ros.TransformException) as error:
                self.get_logger().warning(f'Capture rejected: {error}', throttle_duration_sec=2.)

    def save(self):
        camera = self.frames[0]['camera']
        target = Path(self.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, color_bgr=np.stack([f['color'] for f in self.frames]),
            depth_m=np.stack([f['depth'] for f in self.frames]), K=camera.K, D=np.asarray(camera.D),
            distortion_model=camera.distortion_model, camera_frame=camera.frame_id,
            color_frame=camera.frame_id, depth_frame=camera.frame_id, base_frame=self.params['base_frame'],
            transforms=np.stack([f['transform'] for f in self.frames]),
            stamps=np.array([f['stamp'] for f in self.frames]),
            color_stamps=np.array([f['color_stamp'] for f in self.frames]),
            empty=np.full(len(self.frames), self.empty), provenance='real_or_user_selected_ros_stream')
        self.get_logger().info(f'Saved {len(self.frames)} paired frames to {target}')


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = CaptureRgbd()
        deadline = time.monotonic() + node.timeout
        while rclpy.ok() and len(node.frames) < node.limit and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.1)
        if len(node.frames) < node.limit:
            raise RuntimeError(f'Capture incomplete: {len(node.frames)}/{node.limit}; check registration/TF/topics')
        node.save()
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
