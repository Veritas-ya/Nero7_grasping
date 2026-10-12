"""Arm mode control via the SDK: drag-teach / normal / status.

The agx_arm_ctrl ROS2 node must NOT be running (can0 is single-master).

  action:=drag    enter zero-force drag-teach mode (leader mode);
                  Ctrl+C exits and restores normal control mode
  action:=normal  restore normal single-arm CAN control mode (exits any
                  teach/linkage mode left over from WiFi app control)
  action:=status  print ctrl_mode and exit

Usage:
    ros2 run nero7_control teach_mode --ros-args -p action:=normal
"""
import signal
import time

import rclpy
from rclpy.node import Node

from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, NeroFW

_CTRL_NAMES = {0: 'STANDBY', 1: 'CAN_CTRL', 2: 'TEACHING', 3: 'ETHERNET',
               4: 'WIFI', 5: 'REMOTE', 6: 'LINKAGE_TEACHING_INPUT',
               7: 'OFFLINE_TRAJECTORY'}


def _connect():
    cfg = create_agx_arm_config(
        robot=ArmModel.NERO,
        firmeware_version=NeroFW.DEFAULT,
        interface='socketcan',
        channel='can0',
    )
    robot = AgxArmFactory.create_arm(cfg)
    robot.connect()
    return robot


def _ctrl_mode(robot):
    st = robot.get_arm_status()
    if st is None:
        return None
    return getattr(st.msg, 'ctrl_mode', None)


def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('teach_mode')
    action = node.declare_parameter('action', 'drag').value

    robot = _connect()
    if action == 'status':
        m = _ctrl_mode(robot)
        node.get_logger().info(f'ctrl_mode: {m} ({_CTRL_NAMES.get(m, "?")})')
        raise SystemExit(0)

    if action == 'normal':
        node.get_logger().info(f'ctrl_mode before: {_ctrl_mode(robot)}')
        while not robot.enable():
            time.sleep(0.05)
        robot.set_follower_mode()  # v>=1.12: follower = 单臂正常模式
        robot.set_normal_mode()    # v<=1.11 兼容（新固件上为 no-op）
        time.sleep(1.0)
        m = _ctrl_mode(robot)
        node.get_logger().info(f'ctrl_mode after: {m} ({_CTRL_NAMES.get(m, "?")})')
        if m != 1:
            node.get_logger().warn('still not in CAN control mode; try again or '
                                   'power-cycle the arm')
        raise SystemExit(0 if m == 1 else 1)

    # drag teach mode
    while not robot.enable():
        time.sleep(0.05)
    node.get_logger().info('entering drag-teach mode ...')
    robot.set_leader_mode()

    def _on_sigint(signum, frame):
        node.get_logger().info('exiting teach mode ...')
        robot.set_follower_mode()
        robot.set_normal_mode()
        node.get_logger().info('normal control mode restored')
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _on_sigint)
    signal.signal(signal.SIGTERM, _on_sigint)
    node.get_logger().info('=== DRAG-TEACH ACTIVE: move the arm by hand. '
                           'Ctrl+C to exit ===')
    try:
        while rclpy.ok():
            js = robot.get_joint_angles()
            js = js.msg if js is not None else []
            node.get_logger().info(f'joints: {[round(v, 2) for v in js]}',
                                   throttle_duration_sec=2.0)
            time.sleep(0.2)
    except SystemExit:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
