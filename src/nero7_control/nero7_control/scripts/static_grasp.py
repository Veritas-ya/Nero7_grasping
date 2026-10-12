"""First static grasp test: detect a block, grasp it, drop it at a target.

Uses the calibrated chain: block_detector (with d2c correction) -> block
position in base_link -> arm moves the gripper (gripper_tcp + grasp_rpy from
calibration.yaml) above the block, descends, grips, lifts, drops.

Usage (camera + driver + block_detector running, calibration done):
    ros2 run nero7_control static_grasp --ros-args -p dry_run:=false
"""
import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
import yaml

from nero7_msgs.msg import BlockTrackArray
from nero7_control.arm_client import RosTopicArmClient
from nero7_control.belt import BeltModel
from nero7_control.visualization import SpeedImageViewer

DEFAULT_CALIB = ('/home/ubuntu22/Nero7_grasping/src/nero7_bringup/config/'
                 'calibration.yaml')


class StaticGraspNode(Node):
    def __init__(self):
        super().__init__('static_grasp')
        self._dry_run = self.declare_parameter('dry_run', False).value
        self._descend_offset = self.declare_parameter('descend_offset', 0.025).value
        self._approach_height = self.declare_parameter('approach_height', 0.08).value
        self._lift_height = self.declare_parameter('lift_height', 0.12).value
        self._grip_width = self.declare_parameter('grip_width', 0.030).value
        self._grip_force = self.declare_parameter('grip_force', 2.0).value
        drop = self.declare_parameter('drop_pose', [0.30, 0.0, 0.20]).value
        self._drop = list(map(float, drop))

        # calibration
        self._gripper_tcp = np.zeros(3)
        self._grasp_rpy = [math.pi, 0.0, 0.0]
        try:
            with open(DEFAULT_CALIB) as f:
                data = yaml.safe_load(f) or {}
            if data.get('gripper_tcp'):
                self._gripper_tcp = np.asarray(data['gripper_tcp'], float)
            if data.get('grasp_rpy'):
                self._grasp_rpy = data['grasp_rpy']
        except OSError as e:
            self.get_logger().warn(f'no calibration file: {e}')
        self.get_logger().info(f'gripper_tcp={np.round(self._gripper_tcp * 1000, 1)} '
                               f'mm grasp_rpy={np.round(self._grasp_rpy, 3)}')

        self._arm = RosTopicArmClient(self)
        self._tracks = []
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(BlockTrackArray, '/perception/tracks',
                                 self._tracks_cb, qos)
        self._timer = self.create_timer(1.0, self._run_once)
        self._speed_viewer = None
        enabled = self.declare_parameter('visualization_enabled', True).value
        topic = self.declare_parameter('visualization_topic', '/perception/debug_image').value
        if enabled:
            self._speed_viewer = SpeedImageViewer(self.get_logger(), topic)
            self._viewer_timer = self.create_timer(1.0, self._speed_viewer.check)

    def destroy_node(self):
        if getattr(self, '_speed_viewer', None) is not None:
            self._speed_viewer.close()
        return super().destroy_node()

    def _tracks_cb(self, msg):
        self._tracks = [t for t in msg.tracks]

    def _fingertip_to_flange(self, x, y, z):
        """Flange pose for fingertip at (x,y,z) with the grasp orientation."""
        from scipy.spatial.transform import Rotation as R
        Rf = R.from_euler('xyz', self._grasp_rpy).as_matrix()
        t = np.array([x, y, z]) - Rf @ self._gripper_tcp
        return [t[0], t[1], t[2], self._grasp_rpy[0], self._grasp_rpy[1],
                self._grasp_rpy[2]]

    def _run_once(self):
        if not self._tracks:
            return
        self._timer.cancel()
        self._timer = None
        tr = max(self._tracks, key=lambda t: t.height)  # tallest block
        self.get_logger().info(f'grasping block id={tr.id} at '
                               f'({tr.centroid_base.x:.3f}, {tr.centroid_base.y:.3f}), '
                               f'height={tr.height * 1000:.0f}mm')
        bx, by = tr.centroid_base.x, tr.centroid_base.y
        belt_z = tr.belt_z
        top = belt_z + tr.height

        # 1) open gripper, approach from above
        self._arm.move_gripper(0.060, force=1.0)
        time.sleep(0.8)
        above = self._fingertip_to_flange(bx, by, top + self._approach_height)
        self.get_logger().info(f'approach: {[round(v, 3) for v in above]}')
        self._arm.move_p(above, wait=True, timeout=10.0)
        time.sleep(0.5)

        # 2) descend: fingertips into the block by descend_offset
        grasp_z = top - self._descend_offset
        grasp = self._fingertip_to_flange(bx, by, grasp_z)
        self.get_logger().info(f'descend: {[round(v, 3) for v in grasp]}')
        if not self._dry_run:
            self._arm.move_l(grasp, wait=True, timeout=10.0)
            time.sleep(0.5)

        # 3) grip
        if not self._dry_run:
            self._arm.move_gripper(self._grip_width, force=self._grip_force)
            time.sleep(1.2)
            w = self._arm.get_gripper_width()
            self.get_logger().info(f'gripper width after close: {w}')
            if w is not None and w < self._grip_width * 1.5:
                self.get_logger().info('grasped (width stopped above target)')
            else:
                self.get_logger().warn('possibly empty grasp')

        # 4) lift
        lift = self._fingertip_to_flange(bx, by, top + self._lift_height)
        self._arm.move_l(lift, wait=True, timeout=10.0)
        time.sleep(0.5)

        # 5) drop
        drop = self._fingertip_to_flange(*self._drop)
        self._arm.move_p(drop, wait=True, timeout=10.0)
        time.sleep(0.5)
        self._arm.move_gripper(0.060, force=1.0)
        self.get_logger().info('GRASP CYCLE DONE')
        raise SystemExit(0)


def main(args=None):
    rclpy.init(args=args)
    node = StaticGraspNode()
    try:
        rclpy.spin(node)
    except (SystemExit, KeyboardInterrupt):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
