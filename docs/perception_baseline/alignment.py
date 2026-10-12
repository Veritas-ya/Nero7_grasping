"""RGB-D alignment helpers.

The Gemini 335L color stream (1280x720) and depth stream (848x480) have
different resolutions.  Two ways to get depth on the color grid:

1. ``AlignedTopicSource``: the orbbec driver with ``depth_registration:=true``
   publishes a registered depth image on the color grid (topic confirmed in M0,
   default ``/camera/aligned_depth_to_color/image_raw``).

2. ``PointsSource``: project ``/camera/depth/points`` onto the color grid using
   the static TF between the depth and color optical frames (fallback when the
   driver does not provide an aligned topic).
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from cv_bridge import CvBridge
import tf2_ros
from tf2_ros import TransformException

from .pcl_utils import read_points_xyz


class DepthSource:
    """Base class: yields (depth_color_grid_m, stamp) on the color grid."""

    def latest(self):
        """Return (depth float32 HxW in meters on color grid, stamp) or None."""
        raise NotImplementedError


class AlignedTopicSource(DepthSource):
    """Subscribe to a driver-provided aligned depth topic (mm, on color grid)."""

    def __init__(self, node: Node, topic: str, reliability: str = 'best_effort'):
        self._bridge = CvBridge()
        self._depth = None
        self._stamp = None
        qos = QoSProfile(depth=5)
        if reliability == 'best_effort':
            qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self._sub = node.create_subscription(Image, topic, self._cb, qos)

    def _cb(self, msg: Image):
        img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        if img.dtype == np.uint16:
            img = img.astype(np.float32) / 1000.0  # mm -> m
        else:
            img = img.astype(np.float32)
        img[img <= 0] = np.nan
        self._depth, self._stamp = img, msg.header.stamp

    def latest(self):
        if self._depth is None:
            return None
        return self._depth, self._stamp


class PointsSource(DepthSource):
    """Project the depth point cloud onto the color grid via TF.

    Keeps the nearest point per color pixel.  ~407k points at 30 Hz is heavy in
    pure Python; this runs on a timer and is intended as a fallback, so the
    projection is throttled by ``process_every`` frames / a max rate.
    """

    def __init__(self, node: Node, points_topic: str, color_info_topic: str,
                 target_frame: str, reliability: str = 'best_effort',
                 max_rate: float = 10.0):
        self._node = node
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, node)
        self._target_frame = target_frame
        self._K = None
        self._wh = None
        self._cloud = None
        self._cloud_stamp = None
        self._min_dt = 1.0 / max_rate
        self._last_t = -1e9
        qos = QoSProfile(depth=5)
        if reliability == 'best_effort':
            qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self._sub_points = node.create_subscription(
            PointCloud2, points_topic, self._cloud_cb, qos)
        # camera_info QoS on orbbec driver is sensor_data (best_effort)
        self._sub_info = node.create_subscription(
            CameraInfo, color_info_topic, self._info_cb, qos)

    def _cloud_cb(self, msg: PointCloud2):
        self._cloud, self._cloud_stamp = msg, msg.header.stamp

    def _info_cb(self, msg: CameraInfo):
        self._K = np.array(msg.k, float).reshape(3, 3)
        self._wh = (msg.width, msg.height)

    def _project(self):
        """Project latest cloud onto color grid; return aligned depth image."""
        if self._cloud is None or self._K is None:
            return None, None
        try:
            t = self._tf_buffer.lookup_transform(
                self._target_frame, self._cloud.header.frame_id,
                rclpy.time.Time.from_msg(self._cloud_stamp),
                timeout=rclpy.duration.Duration(seconds=0.1))
        except TransformException as e:
            self._node.get_logger().debug(f'TF lookup failed: {e}')
            return None, None
        from tf2_geometry_msgs import do_transform_point3
        from geometry_msgs.msg import PointStamped
        pts = read_points_xyz(self._cloud)  # Nx3 in cloud frame
        if pts.size == 0:
            return None, None
        # vectorized transform to color optical frame
        q = t.transform.rotation
        R = _quat_to_R(q.w, q.x, q.y, q.z)
        tvec = np.array([t.transform.translation.x, t.transform.translation.y,
                         t.transform.translation.z])
        p = (R @ pts.T).T + tvec
        z = p[:, 2]
        ok = (z > 0.05) & np.isfinite(z)
        p, z = p[ok], z[ok]
        if p.size == 0:
            return None, None
        u = (p[:, 0] / z) * self._K[0, 0] + self._K[0, 2]
        v = (p[:, 1] / z) * self._K[1, 1] + self._K[1, 2]
        w, h = self._wh
        inb = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u, v, z = u[inb].astype(int), v[inb].astype(int), z[inb]
        if z.size == 0:
            return None, None
        # keep nearest depth per pixel
        depth = np.full((h, w), np.nan, np.float32)
        flat = v * w + u
        order = np.argsort(flat, kind='stable')
        flat_s, z_s = flat[order], z[order]
        first = np.empty(flat_s.size, bool)
        first[0] = True
        np.not_equal(flat_s[1:], flat_s[:-1], out=first[1:])
        depth.flat[flat_s[first]] = z_s[first]
        return depth, self._cloud_stamp

    def latest(self):
        now = self._node.get_clock().now()
        t = now.nanoseconds * 1e-9
        if t - self._last_t < self._min_dt:
            return None
        depth, stamp = self._project()
        if depth is not None:
            self._last_t = t
        return depth, stamp


def _quat_to_R(w, x, y, z):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def make_depth_source(node: Node, params) -> DepthSource:
    """Create the configured depth source."""
    mode = params['alignment_mode'] if 'alignment_mode' in params else 'aligned_topic'
    if mode == 'points':
        return PointsSource(
            node, params['points_topic'], params['color_camera_info'],
            params['camera_optical_frame'], params.get('qos_reliability', 'best_effort'))
    return AlignedTopicSource(node, params['depth_topic'],
                              params.get('qos_reliability', 'best_effort'))
