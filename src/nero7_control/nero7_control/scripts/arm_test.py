"""M0 arm verification script: enable -> home -> move_p -> gripper test.

The move_p test uses the current flange pose shifted down by 5 cm (always
reachable), avoiding hard-coded poses that may be outside the arm's IK
envelope.

Usage (with the agx_arm_ctrl driver running):
    ros2 run nero7_control arm_test --ros-args -p mode:=basic
    ros2 run nero7_control arm_test --ros-args -p mode:=gripper

Or against the SDK directly (driver node must NOT be running):
    ros2 run nero7_control arm_test --ros-args -p backend:=sdk
"""
import time

import rclpy
from rclpy.node import Node
import tf2_ros
from tf2_ros import TransformException

from nero7_control.arm_client import RosTopicArmClient, SdkArmClient


class ArmTestNode(Node):
    def __init__(self):
        super().__init__('arm_test')
        backend = self.declare_parameter('backend', 'ros_topic').value
        self._mode = self.declare_parameter('mode', 'basic').value
        self._arm = (SdkArmClient() if backend == 'sdk'
                     else RosTopicArmClient(self))
        self._tf = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf, self)
        self._timer = self.create_timer(0.5, self._run)

    def _run(self):
        self._timer.cancel()
        self.get_logger().info('enabling arm ...')
        self._arm.enable()
        time.sleep(2.0)
        self.get_logger().info('moving home ...')
        self._arm.home()
        self._wait_motion(10.0)
        if self._mode == 'basic':
            self._test_move_p()
        elif self._mode == 'gripper':
            self._test_gripper()
        raise SystemExit(0)

    def _test_move_p(self):
        pose = self._flange_pose()
        if pose is None:
            self.get_logger().error('cannot read flange TF; move_p test skipped')
            return
        x, y, z, qx, qy, qz, qw = pose
        target = [x, y, z - 0.05, qx, qy, qz, qw]  # 5 cm down, same orientation
        self.get_logger().info(f'flange at home: ({x:.3f}, {y:.3f}, {z:.3f})')
        self.get_logger().info(f'move_p to z-0.05: ({x:.3f}, {y:.3f}, {z - 0.05:.3f})')
        self._arm.move_p_quat(target, wait=True, timeout=10.0)
        time.sleep(1.0)
        js = self._arm.get_joint_state()
        self.get_logger().info(f'joint state after move_p: '
                               f'{[round(v, 3) for v in (js or [])]}')
        self.get_logger().info('moving home again ...')
        self._arm.home()
        self._wait_motion(10.0)
        self.get_logger().info('M0 basic arm test DONE')

    def _test_gripper(self):
        self.get_logger().info('open gripper 70mm')
        self._arm.move_gripper(0.070, wait=True)
        time.sleep(1.0)
        self.get_logger().info('close gripper 10mm')
        self._arm.move_gripper(0.010, wait=True)
        time.sleep(1.0)
        w = self._arm.get_gripper_width()
        self.get_logger().info(f'gripper feedback width: {w}')
        self.get_logger().info('M0 gripper test DONE')

    def _flange_pose(self):
        """Current flange (link7) pose in base frame as [x,y,z,qx,qy,qz,qw]."""
        try:
            tf = self._tf.lookup_transform(
                'base_link', 'link7', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0))
        except TransformException as e:
            self.get_logger().warn(f'TF lookup failed: {e}')
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        return [t.x, t.y, t.z, q.x, q.y, q.z, q.w]

    def _wait_motion(self, timeout):
        t0 = time.time()
        # wait for the motion to start (status leaves 0), then to finish
        started = False
        while time.time() - t0 < timeout:
            done = self._arm.is_motion_done()
            if not done:
                started = True
            if started and done:
                return
            time.sleep(0.05)


def main(args=None):
    rclpy.init(args=args)
    node = ArmTestNode()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
