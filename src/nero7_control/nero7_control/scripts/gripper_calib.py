"""AGX gripper teaching calibration (port of pyAgxArm test1.py flow).

Steps (interactive):
1. disable the gripper
2. set max_range_config to the actual stroke (default 0.07 m)
3. you manually squeeze the gripper fully closed
4. run calibrate_gripper()
5. verify by opening/closing

Usage:
    ros2 run nero7_control gripper_calib --ros-args -p backend:=sdk
    ros2 run nero7_control gripper_calib --ros-args -p backend:=ros_topic -p max_range:=0.07

Note: the ROS-topic backend cannot run the teaching calibration (the driver
node owns the CAN bus); use it only to verify open/close after calibrating.
"""
import time

import rclpy
from rclpy.node import Node

from nero7_control.arm_client import RosTopicArmClient, SdkArmClient


def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('gripper_calib')
    backend = node.declare_parameter('backend', 'sdk').value
    max_range = node.declare_parameter('max_range', 0.07).value

    if backend != 'sdk':
        node.get_logger().warn('ros_topic backend: only open/close verification')
        arm = RosTopicArmClient(node)
        arm.enable()
        time.sleep(2.0)
        node.get_logger().info('open 70mm ...')
        arm.move_gripper(0.070, wait=True)
        time.sleep(1.0)
        node.get_logger().info('close 10mm ...')
        arm.move_gripper(0.010, wait=True)
        time.sleep(1.0)
        node.get_logger().info(f'feedback width: {arm.get_gripper_width()}')
        raise SystemExit(0)

    arm = SdkArmClient()
    robot = arm._robot
    robot.connect()
    while not robot.enable():
        time.sleep(0.05)
    gripper = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)

    print('=== AGX gripper teaching calibration ===')
    print(f'1. disable gripper, set max stroke to {max_range} m')
    gripper.disable_gripper()
    gripper.set_gripper_teaching_pendant_param(max_range_config=max_range)
    input('2. Manually squeeze the gripper fully closed, then press Enter ...')
    print('3. calibrating ...')
    gripper.calibrate_gripper()
    print('4. done. Opening to 70mm ...')
    robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.P)
    gripper.move_gripper_m(0.070)
    time.sleep(1.0)
    print('calibration finished')
    raise SystemExit(0)


if __name__ == '__main__':
    main()
