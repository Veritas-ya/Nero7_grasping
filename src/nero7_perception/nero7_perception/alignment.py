"""Registered depth buffering and true point-cloud-to-color reprojection."""
from collections import deque
from dataclasses import dataclass
import numpy as np
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from cv_bridge import CvBridge
import tf2_ros
from tf2_ros import TransformException

from .geometry import CameraGeometry, depth_metres, check_transform
from .pcl_utils import read_points_xyz


def stamp_seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


@dataclass
class DepthFrame:
    depth: np.ndarray
    stamp: object
    frame_id: str
    camera: object = None


class DepthSource:
    def __init__(self, node, buffer_size=8, on_update=None):
        self._node = node
        self._frames = deque(maxlen=buffer_size)
        self._on_update = on_update

    def _append(self, depth, stamp, frame_id, camera=None):
        if self._frames and stamp_seconds(stamp) <= stamp_seconds(self._frames[-1].stamp):
            return
        self._frames.append(DepthFrame(depth, stamp, frame_id, camera))
        if self._on_update is not None:
            self._on_update()

    def closest(self, stamp, max_delta, after=-np.inf):
        frames = [frame for frame in self._frames if stamp_seconds(frame.stamp) > after]
        if not frames:
            return None
        target = stamp_seconds(stamp)
        frame = min(frames, key=lambda f: abs(stamp_seconds(f.stamp) - target))
        return frame if abs(stamp_seconds(frame.stamp) - target) <= max_delta else None

    def latest(self):
        if not self._frames:
            return None
        frame = self._frames[-1]
        return frame.depth, frame.stamp

    def check_camera(self, camera, frame):
        camera.check_grid(frame.depth, frame.frame_id)
        depth_camera = frame.camera
        if depth_camera is None:
            raise ValueError('Registered depth CameraInfo is missing')
        if (depth_camera.frame_id != camera.frame_id
                or (depth_camera.width, depth_camera.height) != (camera.width, camera.height)
                or not np.allclose(depth_camera.K, camera.K, atol=1e-4, rtol=0)
                or len(depth_camera.D) != len(camera.D)
                or not np.allclose(depth_camera.D, camera.D, atol=1e-6, rtol=0)
                or depth_camera.distortion_model != camera.distortion_model):
            raise ValueError('Registered depth intrinsics differ from color; select a genuinely aligned stream')


class AlignedTopicSource(DepthSource):
    def __init__(self, node, topic, reliability='best_effort', buffer_size=8, on_update=None,
                 info_topic='/camera/depth/camera_info', rectified=False):
        super().__init__(node, buffer_size, on_update)
        self._bridge = CvBridge()
        self._camera, self._rectified = None, rectified
        qos = QoSProfile(depth=buffer_size)
        if reliability == 'best_effort':
            qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self._sub = node.create_subscription(Image, topic, self._cb, qos)
        self._sub_info = node.create_subscription(CameraInfo, info_topic, self._info_cb, qos)

    def _info_cb(self, msg):
        try:
            self._camera = CameraGeometry.from_info(msg, self._rectified)
        except ValueError as error:
            self._camera = None
            self._node.get_logger().warning(str(error), throttle_duration_sec=2.)

    def _cb(self, msg):
        try:
            image = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            depth = depth_metres(image, msg.encoding)
        except (ValueError, TypeError) as error:
            self._node.get_logger().warning(str(error), throttle_duration_sec=2.)
            return
        self._append(depth, msg.header.stamp, msg.header.frame_id, self._camera)


def project_cloud(points, transform, camera):
    """Nearest optical Z per rounded color pixel, including distortion."""
    transform = check_transform(transform)
    points = np.asarray(points, float).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    points = points @ transform[:3, :3].T + transform[:3, 3]
    points = points[np.isfinite(points).all(axis=1) & (points[:, 2] > .05)]
    depth = np.full((camera.height, camera.width), np.inf, np.float32)
    if len(points):
        projection = camera.project(points)
        good = (np.isfinite(projection).all(axis=1) & (projection[:, 0] >= -.5)
                & (projection[:, 0] <= camera.width-.5) & (projection[:, 1] >= -.5)
                & (projection[:, 1] <= camera.height-.5))
        points, projection = points[good], projection[good]
        pixels = np.rint(projection).astype(np.int64)
        u, v = pixels.T
        inside = (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
        np.minimum.at(depth.ravel(), v[inside] * camera.width + u[inside], points[inside, 2])
    depth[~np.isfinite(depth)] = np.nan
    return depth


class PointsSource(DepthSource):
    def __init__(self, node, points_topic, color_info_topic, target_frame,
                 reliability='best_effort', max_rate=30., buffer_size=8,
                 on_update=None, rectified=False):
        super().__init__(node, buffer_size, on_update)
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, node)
        self._target_frame, self._rectified = target_frame, rectified
        self._camera = None
        self._min_dt, self._last_t = 1. / max_rate, -np.inf
        qos = QoSProfile(depth=buffer_size)
        if reliability == 'best_effort':
            qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self._sub_points = node.create_subscription(PointCloud2, points_topic, self._cloud_cb, qos)
        self._sub_info = node.create_subscription(CameraInfo, color_info_topic, self._info_cb, qos)

    def _info_cb(self, msg):
        try:
            camera = CameraGeometry.from_info(msg, self._rectified)
            if camera.frame_id != self._target_frame:
                raise ValueError('Color CameraInfo optical frame mismatch')
            self._camera = camera
        except ValueError as error:
            self._camera = None
            self._node.get_logger().warning(str(error), throttle_duration_sec=2.)

    def _cloud_cb(self, msg):
        stamp = stamp_seconds(msg.header.stamp)
        if self._camera is None or stamp - self._last_t < self._min_dt - 1e-6:
            return
        try:
            tf = self._tf_buffer.lookup_transform(
                self._target_frame, msg.header.frame_id, rclpy.time.Time.from_msg(msg.header.stamp),
                timeout=rclpy.duration.Duration(seconds=.05))
            q = tf.transform.rotation
            transform = np.eye(4)
            transform[:3, :3] = _quat_to_R(q.w, q.x, q.y, q.z)
            transform[:3, 3] = [tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z]
            depth = project_cloud(read_points_xyz(msg), transform, self._camera)
        except (TransformException, ValueError) as error:
            self._node.get_logger().warning(f'Point-cloud registration failed: {error}', throttle_duration_sec=2.)
            return
        self._last_t = stamp
        self._append(depth, msg.header.stamp, self._target_frame, self._camera)


def _quat_to_R(w, x, y, z):
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def make_depth_source(node, params, on_update=None):
    options = dict(buffer_size=params.get('sync_buffer_size', 8), on_update=on_update)
    reliability = params.get('qos_reliability', 'best_effort')
    if params['alignment_mode'] == 'points':
        return PointsSource(node, params['points_topic'], params['color_camera_info'],
                            params['camera_optical_frame'], reliability,
                            rectified=params.get('color_rectified', False), **options)
    if params['alignment_mode'] != 'aligned_topic':
        raise ValueError('alignment_mode must be aligned_topic or points')
    return AlignedTopicSource(node, params['depth_topic'], reliability,
                              info_topic=params['depth_camera_info'],
                              rectified=params.get('color_rectified', False), **options)
