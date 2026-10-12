"""A/B benchmark of two GR-ConvNet checkpoints on live camera data (M2).

For every confirmed track (rate-limited), both models infer on the same crop
around the block centroid.  Per model it records: q score, width_mm, and the
grasp-centroid distance in the image; annotated crops are saved for manual
inspection.

Usage (camera + detector running, belt stopped, blocks on the belt):
    ros2 run nero7_control benchmark_models --ros-args \
        -p model_a:=/path/to/jacquard-rgbd-grconvnet3-drop0-ch32 \
        -p model_b:=/path/to/cornell-randsplit-rgbd-grconvnet3-drop1-ch16 \
        -p out_dir:=/tmp/grasp_benchmark
"""
import os
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from nero7_msgs.msg import BlockTrackArray
from cv_bridge import CvBridge
import tf2_ros
from tf2_ros import TransformException

from nero7_control.grasp_inference import GraspInference, make_crop


class BenchmarkNode(Node):
    def __init__(self):
        super().__init__('benchmark_models')
        model_a = self.declare_parameter(
            'model_a',
            '/home/ubuntu22/grasp/robotic-grasping/trained-models/jacquard-rgbd-grconvnet3-drop0-ch32').value
        model_b = self.declare_parameter(
            'model_b',
            '/home/ubuntu22/grasp/robotic-grasping/trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch16').value
        self._out_dir = self.declare_parameter('out_dir', '/tmp/grasp_benchmark').value
        self._base = self.declare_parameter('base_frame', 'base_link').value
        self._cam = self.declare_parameter('camera_optical_frame',
                                           'camera_color_optical_frame').value
        os.makedirs(self._out_dir, exist_ok=True)

        self._models = {
            'A': GraspInference(model_a, max_rate=5.0),
            'B': GraspInference(model_b, max_rate=5.0),
        }
        self._stats = {'A': [], 'B': []}
        self._save_i = 0
        self._last_req = 0.0

        self._bridge = CvBridge()
        self._tf = tf2_ros.Buffer()
        tf2_ros.TransformListener(self._tf, self)
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._color = None
        self._depth = None
        self._stamp = None
        self._K = None
        self.create_subscription(Image, '/camera/color/image_raw', self._img_cb, qos)
        self.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw',
                                 self._depth_cb, qos)
        self.create_subscription(CameraInfo, '/camera/color/camera_info',
                                 self._info_cb, qos)
        self.create_subscription(BlockTrackArray, '/perception/tracks',
                                 self._tracks_cb, qos)
        self._timer = self.create_timer(1.0, self._status)
        self.get_logger().info('benchmark ready; press Ctrl+C for the summary')

    def _img_cb(self, msg):
        self._color = self._bridge.imgmsg_to_cv2(msg, 'bgr8')
        self._stamp = msg.header.stamp

    def _depth_cb(self, msg):
        self._depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def _info_cb(self, msg):
        self._K = np.array(msg.k, float).reshape(3, 3)

    def _tracks_cb(self, msg):
        if (self._color is None or self._depth is None or self._K is None
                or self._stamp is None):
            return
        now = time.monotonic()
        if now - self._last_req < 1.0:
            return
        t = next((x for x in msg.tracks if x.status >= 1), None)
        if t is None:
            return
        try:
            tf = self._tf.lookup_transform(
                self._cam, self._base,
                rclpy.time.Time.from_msg(self._stamp),
                timeout=rclpy.duration.Duration(seconds=0.1))
        except TransformException:
            return
        q = tf.transform.rotation
        R = _quat_R(q.w, q.x, q.y, q.z)
        tv = np.array([tf.transform.translation.x, tf.transform.translation.y,
                       tf.transform.translation.z])
        self._last_req = now
        for name, inf in self._models.items():
            inf.request(self._color, self._depth, t.u_px, t.v_px, self._K,
                        t.stamp.sec + t.stamp.nanosec * 1e-9, {'R': R, 't': tv})
        # collect results and save annotated crops
        for name, inf in self._models.items():
            res = inf.latest(max_age=2.0)
            if res is None or res['status'] != 0:
                continue
            du = res['u_px'] - t.u_px
            dv = res['v_px'] - t.v_px
            self._stats[name].append(dict(score=res['score'],
                                          width=res['width_mm'],
                                          d_px=np.hypot(du, dv)))
            self._save_crop(name, t.u_px, t.v_px, res)

    def _save_crop(self, name, u, v, res):
        rgb, _, _, _ = make_crop(self._color, self._depth, u, v)
        img = rgb.copy()
        cy, cx = res['v_px'] - (v - 112), res['u_px'] - (u - 112)
        c = (int(round(cx)), int(round(cy)))
        cv2.circle(img, c, 6, (0, 255, 0), 2)
        ang = res['yaw']
        cv2.line(img, c, (c[0] + int(40 * np.cos(ang)),
                          c[1] + int(40 * np.sin(ang))), (0, 255, 0), 2)
        cv2.putText(img, f'{name} q={res["score"]:.2f} w={res["width_mm"]:.0f}',
                    (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        fn = os.path.join(self._out_dir, f'{self._save_i:04d}_{name}.png')
        cv2.imwrite(fn, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        self._save_i += 1

    def _status(self):
        for name, s in self._stats.items():
            if s:
                self.get_logger().info(
                    f'{name}: n={len(s)} mean_score={np.mean([x["score"] for x in s]):.3f} '
                    f'mean_width={np.mean([x["width"] for x in s]):.1f}mm '
                    f'mean_d_px={np.mean([x["d_px"] for x in s]):.1f}')


def _quat_R(w, x, y, z):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def main(args=None):
    rclpy.init(args=args)
    node = BenchmarkNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node._status()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
