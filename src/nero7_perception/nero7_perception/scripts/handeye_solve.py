"""Solve the eye-to-hand calibration from collected pairs -> calibration.yaml.

Input npz: R_g2b/t_g2b (flange in base), R_t2c/t_t2c (board in camera).
Output: camera_in_base (T_base_cam) written into calibration.yaml.

Usage:
    ros2 run nero7_perception handeye_solve --ros-args \
        -p input:=/tmp/handeye_data.npz \
        -p output:=/home/ubuntu22/Nero7_grasping/src/nero7_bringup/config/calibration.yaml
"""
import os

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
import yaml


class HandEyeSolveNode(Node):
    def __init__(self):
        super().__init__('handeye_solve')
        self._input = self.declare_parameter('input', '/tmp/handeye_data.npz').value
        self._output = self.declare_parameter(
            'output',
            '/home/ubuntu22/Nero7_grasping/src/nero7_bringup/config/calibration.yaml').value
        self.create_timer(0.5, self._run)

    def _run(self):
        data = np.load(self._input)
        R_g2b, t_g2b = data['R_g2b'], data['t_g2b']
        R_t2c, t_t2c = data['R_t2c'], data['t_t2c']
        n = len(R_g2b)
        self.get_logger().info(f'{n} pose pairs loaded')

        # eye-to-hand: camera fixed, board on the gripper
        # X = camera in base frame  (R_base2cam, t_base2cam)
        R_c2b, t_c2b = cv2.calibrateHandEye(
            R_g2b, t_g2b, R_t2c, t_t2c, method=cv2.CALIB_HAND_EYE_TSAI)

        # reprojection error of the solve (base->cam->board vs cam->board)
        errs = []
        for i in range(n):
            R_est = R_c2b.T @ R_g2b[i]          # board in cam via base
            t_est = R_c2b.T @ (t_g2b[i] - t_c2b)
            R_err = R_est.T @ R_t2c[i]
            ang = float(np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1)))
            errs.append((ang * 180 / np.pi, float(np.linalg.norm(t_est - t_t2c[i]))))
        ang_errs = [e[0] for e in errs]
        t_errs = [e[1] for e in errs]
        self.get_logger().info(f'rotation error: mean {np.mean(ang_errs):.3f} deg, '
                               f'max {np.max(ang_errs):.3f} deg')
        self.get_logger().info(f'translation error: mean {np.mean(t_errs) * 1000:.2f} mm, '
                               f'max {np.max(t_errs) * 1000:.2f} mm')

        # quaternion (w,x,y,z)
        q = _R_to_quat(R_c2b)
        out = {'camera_in_base': {'translation': t_c2b.tolist(),
                                  'quaternion': [float(q[0]), float(q[1]),
                                                 float(q[2]), float(q[3])]}}
        os.makedirs(os.path.dirname(self._output), exist_ok=True)
        with open(self._output, 'w') as f:
            yaml.safe_dump(out, f)
        self.get_logger().info(f'written to {self._output}')
        raise SystemExit(0)


def _R_to_quat(R):
    # standard rotation matrix -> quaternion [w, x, y, z]
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
    node = HandEyeSolveNode()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
