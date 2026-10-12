"""Drag-teach calibration with remote record triggers.

The arm runs in zero-force drag mode (SDK direct, driver node stopped): the
user drags the tip to the touch point by hand, says "record" in chat, and the
operator triggers recording by touching /tmp/nero7_record.  Joint angles are
read from the SDK and FK gives the flange pose (no TF needed).

Phases:
  tcp   - touch ONE fixed point with >=4 orientations -> tcp_offset
  board - touch the board corners (>=4) -> T_base_cam

Trigger files (the operator creates/touches them):
  /tmp/nero7_record  record the current pose
  /tmp/nero7_solve   solve with the recorded poses and exit
  /tmp/nero7_quit    exit without solving

The board phase detects the chessboard from the camera at startup (the board
must be visible BEFORE the arm is dragged over it).

Usage:
    ros2 run nero7_perception drag_calib --ros-args -p phase:=tcp
"""
import os
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, NeroFW

from nero7_perception.calib_lib import (DEFAULT_CALIB, detect_board,
                                         pick_touch_order, save_annotated,
                                         solve_board, solve_tcp,
                                         write_calib_file)

TRIGGER_RECORD = '/tmp/nero7_record'
TRIGGER_SOLVE = '/tmp/nero7_solve'
TRIGGER_QUIT = '/tmp/nero7_quit'


def _mtime(path):
    try:
        return os.stat(path).st_mtime
    except OSError:
        return None


class DragCalibNode(Node):
    def __init__(self):
        super().__init__('drag_calib')
        self._phase = self.declare_parameter('phase', 'tcp').value
        self._calib_file = self.declare_parameter('calib_file', DEFAULT_CALIB).value
        self._board = (self.declare_parameter('board_w', 11).value,
                       self.declare_parameter('board_h', 8).value)

        # camera data (board / verify / fingertip phases)
        self._bridge = CvBridge()
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._color = None
        self._depth = None
        self._K = None
        if self._phase in ('board', 'verify', 'fingertip'):
            self.create_subscription(Image, '/camera/color/image_raw',
                                     self._img_cb, qos)
            self.create_subscription(Image, '/camera/depth/image_raw',
                                     self._depth_cb, qos)
            self.create_subscription(CameraInfo, '/camera/color/camera_info',
                                     self._info_cb, qos)

        # existing tcp offset
        self._tcp = np.zeros(3)
        self._d2c = (0, 0)
        try:
            import yaml
            with open(self._calib_file) as f:
                data = yaml.safe_load(f) or {}
            if data.get('tcp_offset'):
                self._tcp = np.asarray(data['tcp_offset'], float)
            if data.get('d2c_offset'):
                self._d2c = (int(data['d2c_offset'][0]),
                             int(data['d2c_offset'][1]))
        except OSError:
            pass

        # SDK connection
        cfg = create_agx_arm_config(
            robot=ArmModel.NERO, firmeware_version=NeroFW.DEFAULT,
            interface='socketcan', channel='can0')
        self._robot = AgxArmFactory.create_arm(cfg)
        self._robot.connect()
        while not self._robot.enable():
            time.sleep(0.05)

    def _img_cb(self, msg):
        self._color = self._bridge.imgmsg_to_cv2(msg, 'bgr8')

    def _depth_cb(self, msg):
        self._depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def _info_cb(self, msg):
        self._K = np.array(msg.k, float).reshape(3, 3)

    def _flange_T(self):
        ja = self._robot.get_joint_angles()
        if ja is None:
            return None
        pose = self._robot.fk(ja.msg)
        from scipy.spatial.transform import Rotation as R
        T = np.eye(4)
        T[:3, :3] = R.from_euler('xyz', pose[3:6]).as_matrix()
        T[:3, 3] = pose[:3]
        return T

    def _tip(self):
        T = self._flange_T()
        if T is None:
            return None
        return T[:3, :3] @ self._tcp + T[:3, 3]

    def _detect_board_once(self):
        self.get_logger().info('detecting chessboard (move the arm out of '
                               'the way if needed) ...')
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.2)
            if self._color is not None and self._depth is not None and self._K is not None:
                pts3d, pts2d, orig_idx = detect_board(self._color, self._depth,
                                                      self._K, self._board,
                                                      d2c_offset=self._d2c)
                if pts3d is not None:
                    order = pick_touch_order(orig_idx, self._board)
                    save_annotated(self._color, pts2d, order)
                    self.get_logger().info(
                        f'board detected: {len(pts3d)} corners, touch order '
                        f'{order}, annotated image at /tmp/touch_corners.png')
                    return pts3d, pts2d, order, orig_idx
            time.sleep(0.5)

    # ---------------- phases ----------------

    def run(self):
        self.get_logger().info(f'drag_calib phase={self._phase} tcp={self._tcp}')
        if self._phase == 'verify':
            self._run_verify()
            return
        if self._phase == 'fingertip':
            self._run_fingertip()
            return

        # enter drag-teach mode first so the user can move the arm out of the
        # camera view during board detection
        self._robot.set_leader_mode()
        self.get_logger().info('=== 拖拽示教已开启：用手拖动机械臂 ===')

        pts3d_cam = None
        touch_order = None
        if self._phase == 'board':
            pts3d_cam, pts2d, touch_order, _ = self._detect_board_once()
            print(f'板角点编号见 /tmp/touch_corners.png（黄色大圈 T0..T4），'
                  f'按顺序触碰：{touch_order}')
        else:
            print('TCP 阶段：把探针尖触碰同一个固定尖点，换 4 个姿态，')
        print('拖到位置后稳住，在聊天里说“记录”；凑齐后说“求解”。')

        poses, pts_base, pts_cam = [], [], []
        cur = 0
        last_rec = _mtime(TRIGGER_RECORD)
        last_sol = _mtime(TRIGGER_SOLVE)
        last_quit = _mtime(TRIGGER_QUIT)
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.3)
            m_r = _mtime(TRIGGER_RECORD)
            m_s = _mtime(TRIGGER_SOLVE)
            m_q = _mtime(TRIGGER_QUIT)
            if m_q is not None and m_q != last_quit:
                self.get_logger().info('quit trigger')
                break
            if m_r is not None and m_r != last_rec:
                last_rec = m_r
                if self._phase == 'tcp':
                    T = self._flange_T()
                    if T is not None:
                        # reject duplicates (arm has not moved)
                        if poses and np.linalg.norm(T[:3, 3] - poses[-1][:3, 3]) < 0.003:
                            self.get_logger().warn('臂没有移动，与上一记录重复，未记录！'
                                                   '请先拖动机械臂')
                        else:
                            poses.append(T)
                            tip = T[:3, 3]
                            self.get_logger().info(f'记录姿态 {len(poses)}: 法兰='
                                                   f'{np.round(tip * 1000, 1)} mm')
                    else:
                        self.get_logger().warn('无法读取关节角')
                else:
                    tip = self._tip()
                    if tip is not None:
                        if pts_base and np.linalg.norm(tip - pts_base[-1]) < 0.003:
                            self.get_logger().warn('臂没有移动，与上一记录重复，未记录！'
                                                   '请先把臂拖到下一个角点')
                        else:
                            T = self._flange_T()
                            pts_base.append(tip)
                            pts_cam.append(pts3d_cam[touch_order[cur]])
                            self.get_logger().info(
                                f'记录角点 {touch_order[cur]}: tip='
                                f'{np.round(tip * 1000, 1)} mm, 法兰='
                                f'{np.round(T[:3, 3] * 1000, 1)} mm '
                                f'(探针差向量长度 '
                                f'{np.linalg.norm(tip - T[:3, 3]) * 1000:.1f} mm)')
                            cur = min(cur + 1, len(touch_order) - 1)
                            if cur == len(touch_order) - 1:
                                self.get_logger().info('全部角点已记录，可求解')
            if m_s is not None and m_s != last_sol:
                last_sol = m_s
                if self._phase == 'tcp':
                    if len(poses) >= 4:
                        self._solve_tcp(poses)
                        break
                    self.get_logger().warn(f'只有 {len(poses)} 个姿态')
                else:
                    if len(pts_base) >= 4:
                        self._solve_board(pts_base, pts_cam)
                        break
                    self.get_logger().warn(f'只有 {len(pts_base)} 个点')
            tip = self._tip()
            if tip is not None:
                self.get_logger().info(
                    f'tip=({tip[0]:.2f},{tip[1]:.2f},{tip[2]:.2f}) '
                    f'已记录={len(poses) if self._phase == "tcp" else len(pts_base)}',
                    throttle_duration_sec=3.0)
            time.sleep(0.2)

        self._robot.set_follower_mode()  # v>=1.12 退出 leader 的正确方式
        self._robot.set_normal_mode()    # 老固件兼容
        self.get_logger().info('exited drag mode, normal mode restored')

    def _run_verify(self):
        """Holdout verification: touch corners NOT used in the fit, compare
        with the extrinsic prediction."""
        import yaml
        from scipy.spatial.transform import Rotation as R
        self._robot.set_leader_mode()
        self.get_logger().info('=== 拖拽示教已开启：用手拖动机械臂（先把臂移出相机视野）===')
        pts3d, pts2d, _, orig = self._detect_board_once()
        with open(self._calib_file) as f:
            calib = yaml.safe_load(f)
        t = np.array(calib['camera_in_base']['translation'])
        q = calib['camera_in_base']['quaternion']
        Rcb = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        pts_base_pred = (Rcb @ pts3d.T).T + t

        # holdout corners: not [0, 10, 77, 87] (the fit set), spread over the board
        fit = {0, 10, 77, 87}
        holdout = [38, 5, 82, 55]
        holdout_pos = [int(np.where(orig == h)[0][0]) for h in holdout]
        save_annotated(self._color, pts2d, holdout_pos)  # highlighted
        self.get_logger().info(f'留出验证点（黄色圈 T0..T3）：{holdout}，'
                               f'图片 /tmp/touch_corners.png')

        errs = []
        cur = 0
        last_rec = _mtime(TRIGGER_RECORD)
        last_sol = _mtime(TRIGGER_SOLVE)
        last_quit = _mtime(TRIGGER_QUIT)
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.3)
            m_r = _mtime(TRIGGER_RECORD)
            m_s = _mtime(TRIGGER_SOLVE)
            m_q = _mtime(TRIGGER_QUIT)
            if m_q is not None and m_q != last_quit:
                break
            if m_r is not None and m_r != last_rec:
                last_rec = m_r
                tip = self._tip()
                if tip is None:
                    continue
                if cur >= len(holdout):
                    self.get_logger().warn('所有验证点已记录')
                    continue
                i = int(np.where(orig == holdout[cur])[0][0])
                pred = pts_base_pred[i]
                err = np.linalg.norm(tip - pred) * 1000.0
                errs.append(err)
                self.get_logger().info(f'验证点 {holdout[cur]}: 触碰={np.round(tip*1000,1)} mm, '
                                       f'预测={np.round(pred*1000,1)} mm, '
                                       f'误差 {err:.1f} mm')
                cur += 1
            if m_s is not None and m_s != last_sol:
                last_sol = m_s
                if errs:
                    arr = np.array(errs)
                    self.get_logger().info(f'=== 验证结果: mean {arr.mean():.1f} mm, '
                                           f'max {arr.max():.1f} mm (n={len(arr)}) ===')
                    self.get_logger().info('PASS (<10mm)' if arr.max() < 10.0
                                           else 'FAIL (>=10mm)')
                break
            time.sleep(0.2)
        self._robot.set_follower_mode()
        self._robot.set_normal_mode()
        self.get_logger().info('exited drag mode')

    def _run_fingertip(self):
        """Fingertip calibration WITHOUT the board: the user touches the table
        with the closed fingertips at any camera-visible point.  The depth
        camera measures the touch point (largest above-table blob = the
        gripper, at the table plane) and the calibrated extrinsic gives it in
        base -> fingertip offset in the flange frame."""
        import yaml
        from scipy.spatial.transform import Rotation as R
        from nero7_perception.conveyor_model import BeltPlane
        self._robot.set_leader_mode()
        self.get_logger().info('=== 拖拽示教已开启 ===')

        # close the gripper so the fingertips form the touch point
        try:
            gripper = self._robot.init_effector(
                self._robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
            gripper.move_gripper_m(0.0, force=1.0)
            self.get_logger().info('夹爪已合拢')
            time.sleep(0.5)
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f'夹爪合拢失败: {e}')

        with open(self._calib_file) as f:
            calib = yaml.safe_load(f)
        t_cb = np.array(calib['camera_in_base']['translation'])
        q_cb = calib['camera_in_base']['quaternion']
        Rcb = R.from_quat([q_cb[1], q_cb[2], q_cb[3], q_cb[0]]).as_matrix()

        print('用合拢的指尖轻触桌面（相机看得见的任意位置，别碰别的物体），')
        print('碰到后在聊天里说"记录"。')
        last_rec = _mtime(TRIGGER_RECORD)
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.3)
            m_r = _mtime(TRIGGER_RECORD)
            if m_r is not None and m_r != last_rec:
                last_rec = m_r
                T = self._flange_T()
                if T is None:
                    continue
                # measure the touch point from the camera
                fingertip_cam = self._detect_touch_point()
                if fingertip_cam is None:
                    self.get_logger().warn('相机没测到夹爪触地点，再试一次')
                    continue
                fingertip_base = Rcb @ fingertip_cam + t_cb
                Rf, tf = T[:3, :3], T[:3, 3]
                offset = Rf.T @ (fingertip_base - tf)
                rpy = R.from_matrix(Rf).as_euler('xyz')
                self.get_logger().info(f'相机测得触地点 base='
                                       f'{np.round(fingertip_base * 1000, 1)} mm')
                self.get_logger().info(f'gripper_tcp = {np.round(offset * 1000, 1)} mm')
                write_calib_file(self._calib_file, dict(
                    gripper_tcp=offset.tolist(),
                    grasp_rpy=[float(rpy[0]), float(rpy[1]), float(rpy[2])]))
                self.get_logger().info(f'已写入 calibration.yaml (gripper_tcp + '
                                       f'grasp_rpy={np.round(rpy, 3)})')
                break
            time.sleep(0.2)
        self._robot.set_follower_mode()
        self._robot.set_normal_mode()
        self.get_logger().info('exited drag mode')

    def _detect_marker(self, plane_z, depth):
        """HSV color detection for the fingertip marker (default red), with a
        depth check: the marker must be near the table plane (the fingertip
        is touching the table).  Returns the marker position in the camera
        optical frame, or None."""
        import cv2
        if self._color is None or self._K is None:
            return None
        hsv = cv2.cvtColor(self._color, cv2.COLOR_BGR2HSV)
        ranges = {'red': [(0, 70, 50), (12, 255, 255)],
                  'red2': [(168, 70, 50), (180, 255, 255)],
                  'green': [(40, 60, 50), (85, 255, 255)],
                  'blue': [(95, 60, 50), (135, 255, 255)],
                  'yellow': [(20, 60, 60), (35, 255, 255)]}
        color = self.declare_parameter('marker_color', 'red').value
        mask = None
        for name in ([color, color + '2'] if color == 'red' else [color]):
            if name not in ranges:
                continue
            lo, hi = ranges[name]
            m = cv2.inRange(hsv, lo, hi)
            mask = m if mask is None else cv2.bitwise_or(mask, m)
        if mask is None:
            return None
        h, w = mask.shape
        # only the lower part of the image (table region)
        mask[:h // 3, :] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
        if n < 2:
            return None
        # candidates passing the depth check (near the table plane);
        # prefer the LOWEST one (the fingertip marker is on the table)
        best = None
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < 15:
                continue
            u, v = cents[i]
            z_plane = plane_z[int(round(v)), int(round(u))]
            z_meas = float(np.nanmedian(depth[max(0, int(v)) - 3:int(v) + 4,
                                                max(0, int(u)) - 3:int(u) + 4]))
            if not np.isfinite(z_plane) or not np.isfinite(z_meas):
                continue
            if abs(z_meas - z_plane) > 0.08:
                continue  # not touching the table
            if best is None or v > best[2]:
                best = (stats[i, cv2.CC_STAT_AREA], u, v, z_plane, z_meas)
        if best is None:
            return None
        area, u, v, z_plane, z_meas = best
        self.get_logger().info(f'HSV 标记检测: 像素({u:.0f},{v:.0f}) '
                               f'面积={area} 深度差={abs(z_meas - z_plane) * 1000:.0f}mm')
        return np.array([(u - self._K[0, 2]) * z_plane / self._K[0, 0],
                         (v - self._K[1, 2]) * z_plane / self._K[1, 1],
                         z_plane])

    def _detect_touch_point(self):
        """Fingertip position from the colored marker (HSV), falling back to
        the lowest part of the largest above-table blob."""
        import cv2
        from nero7_perception.conveyor_model import BeltPlane
        from nero7_perception.calib_lib import _shift_depth
        if self._depth is None or self._K is None:
            return None
        depth = self._depth.astype(np.float32) / 1000.0
        if self._d2c != (0, 0):
            depth = _shift_depth(depth, *self._d2c)
        h, w = depth.shape
        plane = BeltPlane(distance=0.008, min_ratio=0.25)
        if not plane.fit(depth, roi=(0, h // 3, w, h)):
            return None
        vv, uu = np.mgrid[0:h, 0:w]
        plane_z = (plane.coeffs[0] * uu + plane.coeffs[1] * vv + plane.coeffs[2])

        # 1) marker (HSV + depth check) takes priority
        p = self._detect_marker(plane_z, depth)
        if p is not None:
            return p

        # 2) fallback: lowest part of the largest above-table blob
        height = plane_z - depth
        mask = ((height >= 0.01) & np.isfinite(height)).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
        if n < 2:
            return None
        cands = []
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            med_h = float(np.median(height[labels == i]))
            if area > 1000 and med_h < 0.25:
                cands.append((area, i, cents[i]))
        if not cands:
            return None
        area, i, (u, v) = max(cands, key=lambda c: c[0])
        self.get_logger().info(f'深度启发式: 面积={area}')
        low = (labels == i) & (height < 0.015)
        if low.any():
            ys, xs = np.nonzero(low)
            u = float(np.mean(xs))
            v = float(np.mean(ys))
        z = plane_z[int(round(v)), int(round(u))]
        if not np.isfinite(z):
            return None
        return np.array([(u - self._K[0, 2]) * z / self._K[0, 0],
                         (v - self._K[1, 2]) * z / self._K[1, 1], z])

    def _solve_tcp(self, poses):
        tcp, p_base, resid = solve_tcp(poses)
        self.get_logger().info(f'TCP = {np.round(tcp * 1000, 2)} mm, '
                               f'residual {resid * 1000:.2f} mm')
        write_calib_file(self._calib_file, dict(tcp_offset=tcp.tolist()))
        print(f'TCP 已写入 {self._calib_file}')

    def _solve_board(self, pts_base, pts_cam):
        R, t, resid = solve_board(pts_base, pts_cam)
        from nero7_perception.calib_lib import R_to_quat
        q = R_to_quat(R)
        self.get_logger().info(f'T_base_cam solved: resid mean '
                               f'{resid.mean() * 1000:.1f} mm, '
                               f'max {resid.max() * 1000:.1f} mm')
        write_calib_file(self._calib_file, dict(camera_in_base=dict(
            translation=t.tolist(),
            quaternion=[float(q[0]), float(q[1]), float(q[2]), float(q[3])])))
        print(f'外参已写入 {self._calib_file}')


def main(args=None):
    rclpy.init(args=args)
    node = DragCalibNode()
    try:
        node.run()
    except (SystemExit, KeyboardInterrupt):
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
