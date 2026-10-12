"""Eye-to-hand calibration data collection.

The chessboard is rigidly attached to the arm flange (link7).  The arm visits
a set of poses (interactive or automatic); at each pose the board pose in the
camera is estimated with solvePnP and the flange pose in base_link is read
from TF.  Pairs are saved to an npz for handeye_solve.

Usage (with camera + arm + TF running):
    ros2 run nero7_perception handeye_collect --ros-args \
        -p output:=/tmp/handeye_data.npz -p interactive:=true
    ros2 run nero7_perception handeye_collect --ros-args \
        -p output:=/tmp/handeye_data.npz -p interactive:=false \
        -p poses:="[[-0.2,0.0,0.3,3.14,0.0,0.0], ...]"

Interactive mode: press Enter to capture the current pose, 'q' to finish.
"""
import json
import math
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import tf2_ros
from tf2_ros import TransformException


class HandEyeCollectNode(Node):
    def __init__(self):
        super().__init__('handeye_collect')
        self._output = self.declare_parameter('output', '/tmp/handeye_data.npz').value
        self._interactive = self.declare_parameter('interactive', True).value
        poses_json = self.declare_parameter('poses', '').value
        self._poses = json.loads(poses_json) if poses_json else []
        self._board = (self.declare_parameter('board_w', 11).value,
                       self.declare_parameter('board_h', 8).value)
        self._square = self.declare_parameter('square_m', 0.020).value
        self._flange_frame = self.declare_parameter('flange_frame', 'link7').value
        self._base_frame = self.declare_parameter('base_frame', 'base_link').value

        self._bridge = CvBridge()
        self._tf = tf2_ros.Buffer()
        tf2_ros.TransformListener(self._tf, self)
        self._color = None
        self._stamp = None
        self._K = None
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, '/camera/color/image_raw', self._img_cb, qos)
        self.create_subscription(CameraInfo, '/camera/color/camera_info',
                                 self._info_cb, qos)
        # move the arm through the agx_arm_ctrl topic (no control-pkg dep)
        self._pub_move_p = self.create_publisher(PoseStamped, '/control/move_p', 1)
        self._arm_motion_done = True

        self._R_g2b, self._t_g2b = [], []   # flange in base
        self._R_t2c, self._t_t2c = [], []   # board in camera
        self._criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                          30, 1e-4)
        self._objp = np.zeros((self._board[0] * self._board[1], 3), np.float32)
        self._objp[:, :2] = (np.mgrid[0:self._board[0], 0:self._board[1]]
                             .T.reshape(-1, 2) * self._square)
        self._timer = self.create_timer(0.2, self._step)

    def _img_cb(self, msg):
        self._color = self._bridge.imgmsg_to_cv2(msg, 'bgr8')
        self._stamp = msg.header.stamp

    def _info_cb(self, msg):
        self._K = np.array(msg.k, float).reshape(3, 3)

    def _move_p(self, pose):
        msg = PoseStamped()
        msg.header.frame_id = 'base_link'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = pose[:3]
        cy, sy = math.cos(pose[5] / 2), math.sin(pose[5] / 2)
        cp, sp = math.cos(pose[4] / 2), math.sin(pose[4] / 2)
        cr, sr = math.cos(pose[3] / 2), math.sin(pose[3] / 2)
        msg.pose.orientation.w = cy * cp * cr + sy * sp * sr
        msg.pose.orientation.x = cy * cp * sr - sy * sp * cr
        msg.pose.orientation.y = sy * cp * sr + cy * sp * cr
        msg.pose.orientation.z = sy * cp * cr - cy * sp * sr
        self._pub_move_p.publish(msg)

    def _step(self):
        if self._poses:
            # automatic mode: move through the pose list one by one
            if self._step_idx >= len(self._poses):
                self._save()
                raise SystemExit(0)
            if not self._moved:
                self.get_logger().info(f'move to pose {self._poses[self._step_idx]}')
                self._move_p(self._poses[self._step_idx])
                self._moved = True
                self._move_t0 = time.time()
            elif time.time() - self._move_t0 < 1.5:
                return  # moving + settling
            else:
                self._capture()
                self._step_idx += 1
                self._moved = False

    def run_interactive(self):
        print('=== eye-to-hand collection (interactive) ===')
        print('Move the arm so the chessboard is fully visible, then:')
        print('  Enter -> capture this pose;  q -> save and quit')
        while rclpy.ok():
            cmd = input('> ').strip()
            if cmd == 'q':
                self._save()
                raise SystemExit(0)
            self._capture()
            print(f'captured {len(self._R_g2b)} poses')

    def _capture(self):
        if self._color is None or self._K is None:
            self.get_logger().warn('no camera data yet')
            return
        gray = cv2.cvtColor(self._color, cv2.COLOR_BGR2GRAY)
        ok, corners = cv2.findChessboardCorners(gray, self._board, None)
        if not ok:
            self.get_logger().warn('chessboard not found at this pose')
            return
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                   self._criteria)
        ok, rvec, tvec = cv2.solvePnP(self._objp, corners, self._K, None)
        if not ok:
            self.get_logger().warn('solvePnP failed')
            return
        try:
            tf = self._tf.lookup_transform(
                self._base_frame, self._flange_frame,
                rclpy.time.Time.from_msg(self._stamp),
                timeout=rclpy.duration.Duration(seconds=0.3))
        except TransformException as e:
            self.get_logger().warn(f'flange TF failed: {e}')
            return
        q = tf.transform.rotation
        R = _quat_R(q.w, q.x, q.y, q.z)
        t = np.array([tf.transform.translation.x, tf.transform.translation.y,
                      tf.transform.translation.z])
        Rt, _ = cv2.Rodrigues(rvec)
        self._R_g2b.append(R)
        self._t_g2b.append(t)
        self._R_t2c.append(Rt)
        self._t_t2c.append(tvec.ravel())
        self.get_logger().info(f'captured pose {len(self._R_g2b)}')

    def _save(self):
        if len(self._R_g2b) < 5:
            self.get_logger().warn('too few poses captured')
            return
        np.savez(self._output,
                 R_g2b=np.array(self._R_g2b), t_g2b=np.array(self._t_g2b),
                 R_t2c=np.array(self._R_t2c), t_t2c=np.array(self._t_t2c))
        self.get_logger().info(f'saved {len(self._R_g2b)} poses to {self._output}')


def _quat_R(w, x, y, z):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def main(args=None):
    rclpy.init(args=args)
    node = HandEyeCollectNode()
    node._step_idx = 0
    node._moved = False
    try:
        if node._interactive and not node._poses:
            node.run_interactive()
        rclpy.spin(node)
    except (SystemExit, KeyboardInterrupt):
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
