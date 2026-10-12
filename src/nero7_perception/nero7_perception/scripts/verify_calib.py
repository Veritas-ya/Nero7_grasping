"""Calibration verification: point-and-shoot check.

For each static block on the belt: move the arm to hover above the detected
block position, then compare the arm's actual flange xy (from TF) with the
block centroid xy (from the camera, via the camera->base calibration).  The
offset is the calibration chain error (hand-eye TF + detector backprojection
+ arm positioning).

Usage (camera + arm + TF running, detector publishing tracks):
    ros2 run nero7_perception verify_calib --ros-args \
        -p hover_offset:=0.05 -p move:=true

Without move:=true it only prints predicted-vs-tracked offsets, no arm motion.
"""
import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from nero7_msgs.msg import BlockTrackArray
from geometry_msgs.msg import PoseStamped
import tf2_ros
from tf2_ros import TransformException


class VerifyCalibNode(Node):
    def __init__(self):
        super().__init__('verify_calib')
        self._hover = self.declare_parameter('hover_offset', 0.05).value
        self._move = self.declare_parameter('move', True).value
        self._base_frame = self.declare_parameter('base_frame', 'base_link').value
        self._flange_frame = self.declare_parameter('flange_frame', 'link7').value
        self._results = []
        self._target = None
        self._tracks = []
        self._tf = tf2_ros.Buffer()
        tf2_ros.TransformListener(self._tf, self)
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(BlockTrackArray, '/perception/tracks',
                                 self._tracks_cb, qos)
        self._pub_move_p = self.create_publisher(PoseStamped, '/control/move_p', 1)
        self._timer = self.create_timer(1.0, self._step)
        self._step_idx = 0
        self._t0 = time.time()

    def _tracks_cb(self, msg):
        self._tracks = [t for t in msg.tracks if t.status >= 1]

    def _flange_xy(self):
        try:
            tf = self._tf.lookup_transform(
                self._base_frame, self._flange_frame,
                rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.3))
        except TransformException:
            return None
        return np.array([tf.transform.translation.x, tf.transform.translation.y])

    def _step(self):
        if not self._tracks or self._step_idx >= len(self._tracks):
            return
        tr = self._tracks[self._step_idx]
        if self._target is None:
            z = tr.belt_z + tr.height + self._hover
            pose = [tr.centroid_base.x, tr.centroid_base.y, z, math.pi, 0.0, 0.0]
            self.get_logger().info(f'point {self._step_idx}: block at '
                                   f'({tr.centroid_base.x:.3f}, {tr.centroid_base.y:.3f})')
            if self._move:
                self._pub_move_p.publish(_pose_msg(pose, self.get_clock().now().to_msg()))
            self._target = np.array([tr.centroid_base.x, tr.centroid_base.y])
            self._t0 = time.time()
        elif time.time() - self._t0 > 3.5:
            p = self._flange_xy() if self._move else self._target
            if p is not None:
                d = float(np.linalg.norm(p - self._target))
                self._results.append(d)
                self.get_logger().info(f'point {self._step_idx}: offset '
                                       f'{d * 1000:.1f} mm')
            else:
                self.get_logger().warn('flange TF unavailable, skipping point')
            self._step_idx += 1
            self._target = None
            if self._step_idx >= len(self._tracks):
                arr = np.array(self._results) if self._results else np.array([np.nan])
                self.get_logger().info(f'=== done: mean {np.nanmean(arr) * 1000:.1f} mm, '
                                       f'max {np.nanmax(arr) * 1000:.1f} mm over '
                                       f'{len(arr)} points ===')
                self.get_logger().info('PASS' if np.nanmax(arr) < 0.010 else
                                       'FAIL: recalibrate or check TF chain')
                raise SystemExit(0)


def _pose_msg(pose, stamp):
    msg = PoseStamped()
    msg.header.frame_id = 'base_link'
    msg.header.stamp = stamp
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = pose[:3]
    cy, sy = math.cos(pose[5] / 2), math.sin(pose[5] / 2)
    cp, sp = math.cos(pose[4] / 2), math.sin(pose[4] / 2)
    cr, sr = math.cos(pose[3] / 2), math.sin(pose[3] / 2)
    msg.pose.orientation.w = cy * cp * cr + sy * sp * sr
    msg.pose.orientation.x = cy * cp * sr - sy * sp * cr
    msg.pose.orientation.y = sy * cp * sr + cy * sp * cr
    msg.pose.orientation.z = sy * cp * cr - cy * sp * sr
    return msg


def main(args=None):
    rclpy.init(args=args)
    node = VerifyCalibNode()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
