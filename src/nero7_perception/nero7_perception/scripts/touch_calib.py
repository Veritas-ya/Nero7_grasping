"""Touch-based eye-to-hand calibration: fixed camera + fixed board.

The chessboard is fixed in the workspace (visible to the camera).  The arm's
tool tip (probe held in the gripper, or the gripper fingertips) is jogged to
touch board points; FK + TCP gives those points in base_link.  The same board
corners are measured in the camera frame from the aligned depth image.  The
point correspondences solve T_base_cam by Kabsch (SVD).

Two phases:
  tcp   - 4-point TCP calibration: touch ONE fixed point with >=4 different
          arm orientations -> tool tip offset in the flange frame (written to
          calibration.yaml as tcp_offset)
  board - touch >=4 board corners -> T_base_cam (written to calibration.yaml)

Keyboard (board phase; tcp phase records the same fixed point each time):
  x/X y/Y z/Z  jog the tip +/- step along base axes (motion via /control/move_l)
  a/d w/s q/e  rotate +/- rot_step deg about base Z/Y/X, centered at the tip
               (keeps the tip on the point while the orientation changes)
  [ / ]        halve / double the jog step
  Enter        record the current tip position for the next point in order
  s            solve with the recorded points (>=4)
  q            quit without solving

Requirements while running: arm_only (driver + TF) and the camera running.

Usage:
    ros2 run nero7_perception touch_calib --ros-args -p phase:=tcp
    ros2 run nero7_perception touch_calib --ros-args -p phase:=board
"""
import math
import os
import sys
import termios
import threading
import tty

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
import yaml

DEFAULT_CALIB = ('/home/ubuntu22/Nero7_grasping/src/nero7_bringup/config/'
                 'calibration.yaml')


class TouchCalibNode(Node):
    def __init__(self):
        super().__init__('touch_calib')
        self._phase = self.declare_parameter('phase', 'board').value
        self._board = (self.declare_parameter('board_w', 11).value,
                       self.declare_parameter('board_h', 8).value)
        self._square = self.declare_parameter('square_m', 0.020).value
        self._calib_file = self.declare_parameter('calib_file', DEFAULT_CALIB).value
        self._steps = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
        self._step_idx = 1  # start at 2mm
        self._step = self._steps[self._step_idx]
        self._rot_step = self.declare_parameter('rot_step_deg', 5.0).value
        self._flange_frame = self.declare_parameter('flange_frame', 'link7').value
        self._base_frame = self.declare_parameter('base_frame', 'base_link').value

        # tool tip offset in the flange frame [x,y,z]; from a previous tcp phase
        self._tcp = np.zeros(3)
        self._load_calib_file()

        self._bridge = CvBridge()
        self._tf = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf, self)
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._color = None
        self._depth = None
        self._K = None
        self.create_subscription(Image, '/camera/color/image_raw', self._img_cb, qos)
        self.create_subscription(Image, '/camera/depth/image_raw', self._depth_cb, qos)
        self.create_subscription(CameraInfo, '/camera/color/camera_info',
                                 self._info_cb, qos)
        self._pub_move_l = self.create_publisher(PoseStamped, '/control/move_l', 1)
        from std_srvs.srv import Empty
        self._srv_home = self.create_client(Empty, '/move_home')

        # board geometry
        self._objp = np.zeros((self._board[0] * self._board[1], 3), np.float32)
        self._objp[:, :2] = (np.mgrid[0:self._board[0], 0:self._board[1]]
                             .T.reshape(-1, 2) * self._square)
        self._criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                          30, 1e-4)

    # ---------------- data ----------------

    def _load_calib_file(self):
        try:
            with open(self._calib_file) as f:
                data = yaml.safe_load(f) or {}
            t = data.get('tcp_offset')
            if t:
                self._tcp = np.asarray(t, float)
                self.get_logger().info(f'loaded tcp_offset: {self._tcp}')
        except OSError:
            pass

    def _img_cb(self, msg):
        self._color = self._bridge.imgmsg_to_cv2(msg, 'bgr8')

    def _depth_cb(self, msg):
        self._depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def _info_cb(self, msg):
        self._K = np.array(msg.k, float).reshape(3, 3)

    def _flange_pose(self):
        """Flange (link7) pose in base as 4x4 matrix."""
        try:
            tf = self._tf.lookup_transform(
                self._base_frame, self._flange_frame, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.3))
        except TransformException as e:
            self.get_logger().warn(f'flange TF failed: {e}')
            return None
        q = tf.transform.rotation
        R = _quat_R(q.w, q.x, q.y, q.z)
        t = np.array([tf.transform.translation.x, tf.transform.translation.y,
                      tf.transform.translation.z])
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, t
        return T

    def _tip_in_base(self):
        """Current tool tip position in base frame."""
        T = self._flange_pose()
        if T is None:
            return None
        return T[:3, :3] @ self._tcp + T[:3, 3]

    def _jog(self, axis, sign):
        tip = self._tip_in_base()
        T = self._flange_pose()
        if tip is None or T is None:
            return
        target = T.copy()
        target[:3, 3] = tip
        target[axis, 3] += sign * self._step
        self._move_l_T(target)

    def _rotate_about_tip(self, axis, sign):
        """Rotate the flange pose by rot_step about a base axis through the tip."""
        tip = self._tip_in_base()
        T = self._flange_pose()
        if tip is None or T is None:
            return
        ang = np.deg2rad(self._rot_step) * sign
        v = np.zeros(3)
        v[axis] = 1.0
        R = _axis_angle_R(v, ang)
        # rotate about the tip point: T' = T_rot @ T
        t = tip - R @ tip
        T_rot = np.eye(4)
        T_rot[:3, :3], T_rot[:3, 3] = R, t
        self._move_l_T(T_rot @ T)

    def _move_l_T(self, T):
        msg = PoseStamped()
        msg.header.frame_id = self._base_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = T[:3, 3]
        q = _R_to_quat(T[:3, :3])  # [w,x,y,z]
        msg.pose.orientation.w, msg.pose.orientation.x = q[0], q[1]
        msg.pose.orientation.y, msg.pose.orientation.z = q[2], q[3]
        self._pub_move_l.publish(msg)

    # ---------------- board detection ----------------

    def detect_board(self):
        """Return Nx3 board corner points in the camera optical frame."""
        if self._color is None or self._depth is None or self._K is None:
            return None, None
        gray = cv2.cvtColor(self._color, cv2.COLOR_BGR2GRAY)
        ok, corners = cv2.findChessboardCorners(gray, self._board, None)
        if not ok:
            return None, None
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self._criteria)
        pts2d = corners.reshape(-1, 2)
        # depth at each corner: median over a 7x7 window
        h, w = self._depth.shape[:2]
        pts3d = np.zeros((len(pts2d), 3))
        valid = np.ones(len(pts2d), bool)
        for i, (u, v) in enumerate(pts2d):
            u0, v0 = max(0, int(round(u)) - 3), max(0, int(round(v)) - 3)
            u1, v1 = min(w, u0 + 7), min(h, v0 + 7)
            win = self._depth[v0:v1, u0:u1].astype(np.float32)
            win[win <= 0] = np.nan
            z = float(np.nanmedian(win)) / 1000.0
            if not np.isfinite(z) or z <= 0:
                valid[i] = False
                continue
            pts3d[i] = [(u - self._K[0, 2]) * z / self._K[0, 0],
                        (v - self._K[1, 2]) * z / self._K[1, 1], z]
        pts3d = pts3d[valid]
        if len(pts3d) < 4:
            return None, None
        # all corners are coplanar: robust plane fit, project onto it
        mean = pts3d.mean(axis=0)
        _, _, vh = np.linalg.svd(pts3d - mean)
        n = vh[2]
        pts3d = pts3d - (np.sum((pts3d - mean) * n, axis=1, keepdims=True)) * n
        return pts3d, corners.reshape(-1, 2)[valid]

    def save_annotated(self, pts2d):
        img = self._color.copy()
        for i, (u, v) in enumerate(pts2d):
            cv2.circle(img, (int(round(u)), int(round(v))), 5,
                       (0, 255, 0), 2)
            cv2.putText(img, str(i), (int(round(u)) + 8, int(round(v)) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        path = '/tmp/touch_corners.png'
        cv2.imwrite(path, img)
        self.get_logger().info(f'annotated board saved to {path}')
        return path

    def _go_home(self):
        from std_srvs.srv import Empty
        if self._srv_home.wait_for_service(timeout_sec=1.0):
            self._srv_home.call_async(Empty.Request())
            print('moving home ...')
        else:
            print('/move_home service unavailable')

    def _cycle_step(self, direction):
        self._step_idx = max(0, min(len(self._steps) - 1,
                                    self._step_idx + direction))
        self._step = self._steps[self._step_idx]
        print(f'步长 = {self._step * 1000:.0f} mm')

    # ---------------- phases ----------------

    def run(self):
        self.get_logger().info(f'touch_calib phase={self._phase} '
                               f'tcp={self._tcp} step={self._step * 1000:.1f}mm')
        if self._phase == 'tcp':
            self._run_tcp()
        else:
            self._run_board()

    def _run_tcp(self):
        print('=== TCP 标定（4 点法）===')
        print('把探针尖触碰同一个固定尖点（桌面任一固定尖点，如板角、螺丝尖），')
        print('每次换一个臂的姿态（绕尖点旋转：a/d 绕Z、w/s 绕Y、q/e 绕X），')
        print('单键直发：h 回零位；x/X y/Y z/Z 平移；a/d w/s q/e 旋转；')
        print('[ / ] 步长档位(1,2,5,10,20,50mm)；Enter 记录；s 求解；q 退出。')
        poses = []  # flange 4x4
        while rclpy.ok():
            tip = self._tip_in_base()
            status = (f'尖点=({tip[0]:.3f},{tip[1]:.3f},{tip[2]:.3f}) '
                      f'步长={self._step * 1000:.1f}mm 已记录={len(poses)}'
                      ) if tip is not None else 'TF 不可用'
            sys.stdout.write(f'\r[{status}] > ')
            sys.stdout.flush()
            c = _getch()
            sys.stdout.write('\n')
            if c in ('q', '\x03'):
                raise SystemExit(0)
            if c == 'h':
                self._go_home()
                continue
            if c == 's':
                if len(poses) >= 4:
                    break
                print(f'只有 {len(poses)} 个姿态，至少需要 4 个')
                continue
            if c == '\r':
                T = self._flange_pose()
                if T is not None:
                    poses.append(T)
                    print(f'recorded pose {len(poses)}')
                continue
            if c in 'xXyYzZ':
                axis = 'xyz'.index(c.lower())
                sign = 1 if c.isupper() else -1
                self._jog(axis, sign)
                continue
            if c in 'adwsqe':
                axis = {'a': 2, 'd': 2, 'w': 1, 's': 1, 'q': 0, 'e': 0}[c]
                sign = 1 if c in 'dws' else -1
                self._rotate_about_tip(axis, sign)
                continue
            if c in '[]':
                self._cycle_step(1 if c == ']' else -1)
                continue
        self._solve_tcp(poses)
        raise SystemExit(0)
        self._solve_tcp(poses)
        raise SystemExit(0)

    def _solve_tcp(self, poses):
        """4-point TCP: tip position in flange frame, least squares."""
        A, b = [], []
        for T in poses:
            R, t = T[:3, :3], T[:3, 3]
            A.append(np.hstack([R, -np.eye(3)]))
            b.append(-t)
        A, b = np.vstack(A), np.concatenate(b)
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        tcp, p_base = x[:3], x[3:]
        resid = float(np.linalg.norm(A @ x - b))
        self.get_logger().info(f'TCP offset (flange frame): {np.round(tcp * 1000, 2)} mm')
        self.get_logger().info(f'fit residual: {resid * 1000:.2f} mm')
        self._write_yaml(dict(tcp_offset=tcp.tolist()))
        print(f'TCP = {np.round(tcp * 1000, 1)} mm，已写入 {self._calib_file}')

    def _run_board(self):
        print('=== 触碰式眼在手外标定（固定板）===')
        print('等待棋盘格检测 ...')
        pts3d_cam, pts2d = None, None
        while rclpy.ok() and pts3d_cam is None:
            pts3d_cam, pts2d = self.detect_board()
            if pts3d_cam is None:
                import time
                time.sleep(1.0)
                self.get_logger().info('board not found yet ...')
        self.save_annotated(pts2d)
        n = len(pts3d_cam)
        print(f'检测到 {n} 个角点，编号见 /tmp/touch_corners.png（打开看图）')
        # touch order: 4 extreme corners + center
        idx_center = (self._board[1] - 1) // 2 * self._board[0] + (self._board[0] - 1) // 2
        touch_order = [0, self._board[0] - 1, (self._board[1] - 1) * self._board[0],
                       self._board[0] * self._board[1] - 1, idx_center]
        # filter to detected indices
        touch_order = [i for i in touch_order if i < len(pts3d_cam)]
        pts_base, pts_cam, cur = [], [], 0
        print(f'请按顺序触碰角点：{touch_order}')
        print(f'当前要触碰的是角点 {touch_order[cur]}（见图片编号）')
        print('单键直发：x/X y/Y z/Z 平移；a/d w/s q/e 绕尖点旋转(绕Z/Y/X)；'
              '[ / ] 调步长；Enter 记录；s 求解；q 退出')
        while rclpy.ok():
            tip = self._tip_in_base()
            status = (f'尖点=({tip[0]:.3f},{tip[1]:.3f},{tip[2]:.3f}) '
                      f'步长={self._step * 1000:.1f}mm 已记录={len(pts_base)} '
                      f'待碰角点={touch_order[cur]}') if tip is not None else 'TF 不可用'
            sys.stdout.write(f'\r[{status}] > ')
            sys.stdout.flush()
            c = _getch()
            sys.stdout.write('\n')
            if c in ('q', '\x03'):
                raise SystemExit(0)
            if c == 'h':
                self._go_home()
                continue
            if c == 's':
                if len(pts_base) >= 4:
                    break
                print('至少需要 4 个点')
                continue
            if c in 'xXyYzZ':
                axis = 'xyz'.index(c.lower())
                sign = 1 if c.isupper() else -1
                self._jog(axis, sign)
                continue
            if c in 'adwsqe':
                axis = {'a': 2, 'd': 2, 'w': 1, 's': 1, 'q': 0, 'e': 0}[c]
                sign = 1 if c in 'dws' else -1
                self._rotate_about_tip(axis, sign)
                continue
            if c in '[]':
                self._cycle_step(1 if c == ']' else -1)
                continue
            if c == '\r' and tip is not None:
                pts_base.append(tip)
                pts_cam.append(pts3d_cam[touch_order[cur]])
                print(f'记录角点 {touch_order[cur]}: base={np.round(tip * 1000, 1)} mm')
                cur += 1
                if cur >= len(touch_order):
                    print('全部角点已记录，按 s 求解（或继续 Enter 重录最后一个点）')
                    cur -= 1
                else:
                    print(f'下一个：角点 {touch_order[cur]}')
        self._solve_board(pts_base, pts_cam)
        raise SystemExit(0)

    def _solve_board(self, pts_base, pts_cam):
        pts_base = np.asarray(pts_base)
        pts_cam = np.asarray(pts_cam)
        # Kabsch: T_base_cam  (p_base = R p_cam + t)
        mb, mc = pts_base.mean(axis=0), pts_cam.mean(axis=0)
        H = (pts_cam - mc).T @ (pts_base - mb)
        U, _, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        D = np.diag([1, 1, d])
        R = Vt.T @ D @ U.T
        t = mb - R @ mc
        resid = np.linalg.norm((R @ pts_cam.T).T + t - pts_base, axis=1)
        self.get_logger().info(f'T_base_cam solved with {len(pts_base)} points')
        self.get_logger().info(f'residuals: mean {resid.mean() * 1000:.1f} mm, '
                               f'max {resid.max() * 1000:.1f} mm')
        q = _R_to_quat(R)  # [w,x,y,z]
        self._write_yaml(dict(camera_in_base=dict(
            translation=t.tolist(),
            quaternion=[float(q[0]), float(q[1]), float(q[2]), float(q[3])])))
        print(f'已写入 {self._calib_file}')
        print('下一步：运行 verify_calib 做“指哪打哪”验证（<10mm 通过）')

    def _write_yaml(self, update):
        data = {}
        if os.path.exists(self._calib_file):
            with open(self._calib_file) as f:
                data = yaml.safe_load(f) or {}
        data.update(update)
        os.makedirs(os.path.dirname(self._calib_file), exist_ok=True)
        with open(self._calib_file, 'w') as f:
            yaml.safe_dump(data, f)


def _getch():
    """Read one character in raw mode (no Enter needed)."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _quat_R(w, x, y, z):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _axis_angle_R(v, ang):
    """Rotation matrix for rotation by ang about unit axis v (Rodrigues)."""
    v = np.asarray(v, float)
    v = v / np.linalg.norm(v)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def _R_to_quat(R):
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    y = np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
    z = np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    x *= -1 if R[2, 1] - R[1, 2] < 0 else 1
    y *= -1 if R[0, 2] - R[2, 0] < 0 else 1
    z *= -1 if R[1, 0] - R[0, 1] < 0 else 1
    return np.array([w, x, y, z])


def main(args=None):
    rclpy.init(args=args)
    node = TouchCalibNode()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    try:
        node.run()
    except (SystemExit, KeyboardInterrupt):
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
