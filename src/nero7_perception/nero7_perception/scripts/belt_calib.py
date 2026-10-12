"""Belt direction calibration: track a block on the running belt and fit the
motion direction in base_link, then write belt.yaml.

Usage (camera + TF + detector running, belt moving):
    ros2 run nero7_perception belt_calib --ros-args \
        -p output:=/home/ubuntu22/Nero7_grasping/src/nero7_bringup/config/belt.yaml \
        -p duration:=10.0
"""
import os

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from nero7_msgs.msg import BlockTrackArray
import yaml


class BeltCalibNode(Node):
    def __init__(self):
        super().__init__('belt_calib')
        self._output = self.declare_parameter(
            'output',
            '/home/ubuntu22/Nero7_grasping/src/nero7_bringup/config/belt.yaml').value
        self._duration = self.declare_parameter('duration', 10.0).value
        self._samples = []  # (x, y) in base
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(BlockTrackArray, '/perception/tracks',
                                 self._tracks_cb, qos)
        self.create_timer(1.0, self._finish)
        self.get_logger().info('put a block on the moving belt ...')

    def _tracks_cb(self, msg):
        for t in msg.tracks:
            if t.status >= 1:
                self._samples.append((t.centroid_base.x, t.centroid_base.y))

    def _finish(self):
        if len(self._samples) < 10:
            self.get_logger().info(f'collecting ... {len(self._samples)} samples')
            return
        pts = np.array(self._samples)
        # total-least-squares line fit (PCA)
        mean = pts.mean(axis=0)
        _, _, vh = np.linalg.svd(pts - mean)
        d = vh[0]  # dominant direction
        if d[0] < 0:
            d = -d
        d = d / np.linalg.norm(d)
        resid = np.std((pts - mean) @ vh[1])
        self.get_logger().info(f'bel direction: [{d[0]:.4f}, {d[1]:.4f}], '
                               f'residual std {resid * 1000:.2f} mm')
        # origin: point on the fitted line closest to base origin
        p0 = mean - (mean @ d) * d
        out = {'belt_direction': [float(d[0]), float(d[1])],
               'belt_origin': [float(p0[0]), float(p0[1])],
               'gate_s': 0.35, 'belt_height': 0.0}
        # keep an existing gate_s if the file is already there
        if os.path.exists(self._output):
            with open(self._output) as f:
                old = yaml.safe_load(f) or {}
            if 'gate_s' in old:
                out['gate_s'] = old['gate_s']
        os.makedirs(os.path.dirname(self._output), exist_ok=True)
        with open(self._output, 'w') as f:
            yaml.safe_dump(out, f)
        self.get_logger().info(f'written to {self._output}')
        raise SystemExit(0)


def main(args=None):
    rclpy.init(args=args)
    node = BeltCalibNode()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
