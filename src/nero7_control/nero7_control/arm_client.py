"""Arm abstraction layer with two interchangeable backends.

RosTopicArmClient (default)
    Talks to the agx_arm_ctrl ROS2 node (/control/* topics, /enable_agx_arm,
    /move_home, /emergency_stop services).  Used together with the driver's
    robot_state_publisher for TF.

SdkArmClient (fallback)
    Drives the arm directly over CAN via pyAgxArm.  Mutually exclusive with
    the ROS2 driver node (can0 has a single master): never run both.

Both expose the same ArmClient interface so the FSM does not care which
backend is active.
"""
import math
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import SetBool, Empty


class ArmClient:
    """Common interface; pose = [x, y, z, roll, pitch, yaw] in base frame."""

    def enable(self):
        raise NotImplementedError

    def home(self):
        raise NotImplementedError

    def move_p(self, pose, wait=False, timeout=5.0):
        raise NotImplementedError

    def move_p_quat(self, pose, wait=False, timeout=5.0):
        """move_p with orientation as quaternion: [x,y,z,qx,qy,qz,qw]."""
        raise NotImplementedError

    def move_l(self, pose, wait=False, timeout=5.0):
        raise NotImplementedError

    def move_j(self, joints, wait=False, timeout=5.0):
        raise NotImplementedError

    def move_gripper(self, width_m, force=1.0, wait=False, timeout=3.0):
        raise NotImplementedError

    def get_joint_state(self):
        """Latest 7 arm joint angles (rad), or None."""
        raise NotImplementedError

    def get_gripper_width(self):
        """Latest gripper opening (m), or None."""
        raise NotImplementedError

    def is_motion_done(self):
        """True when the last commanded motion has reached its target."""
        raise NotImplementedError

    def estop(self):
        raise NotImplementedError

    @staticmethod
    def rpy_to_quat(r, p, y):
        cy, sy = math.cos(y / 2), math.sin(y / 2)
        cp, sp = math.cos(p / 2), math.sin(p / 2)
        cr, sr = math.cos(r / 2), math.sin(r / 2)
        return (cy * cp * cr + sy * sp * sr,
                cy * cp * sr - sy * sp * cr,
                sy * cp * sr + cy * sp * cr,
                sy * cp * cr - cy * sp * sr)


class RosTopicArmClient(ArmClient):
    """Backend over the agx_arm_ctrl ROS2 node topics/services."""

    def __init__(self, node: Node, prefix=''):
        self._node = node
        p = prefix
        self._pub_move_p = node.create_publisher(PoseStamped, p + 'control/move_p', 1)
        self._pub_move_l = node.create_publisher(PoseStamped, p + 'control/move_l', 1)
        self._pub_move_j = node.create_publisher(JointState, p + 'control/move_j', 1)
        self._pub_joint = node.create_publisher(JointState, p + 'control/joint_states', 1)
        self._srv_enable = node.create_client(SetBool, p + 'enable_agx_arm')
        self._srv_home = node.create_client(Empty, p + 'move_home')
        self._srv_estop = node.create_client(Empty, p + 'emergency_stop')

        self._joint_state = None
        self._gripper_width = None
        self._motion_done = True
        node.create_subscription(JointState, p + 'feedback/joint_states',
                                 self._joint_cb, 5)
        try:
            from agx_arm_msgs.msg import AgxArmStatus, GripperStatus
            node.create_subscription(AgxArmStatus, p + 'feedback/arm_status',
                                     self._status_cb, 5)
            node.create_subscription(GripperStatus, p + 'feedback/gripper_status',
                                     self._gripper_cb, 5)
        except ImportError:
            node.get_logger().warn('agx_arm_msgs not available; motion status '
                                   'will fall back to timeout')

    # --- feedback ---

    def _joint_cb(self, msg):
        self._joint_state = list(msg.position)

    def _status_cb(self, msg):
        self._motion_done = (msg.motion_status == 0)

    def _gripper_cb(self, msg):
        self._gripper_width = msg.width

    def get_joint_state(self):
        return self._joint_state

    def get_gripper_width(self):
        return self._gripper_width

    def is_motion_done(self):
        return self._motion_done

    # --- commands ---

    def _wait(self, timeout):
        """Wait for the motion to start (status leaves 0), then to finish.

        Without the start phase, a wait issued right after the command would
        return immediately on the stale idle status."""
        if not self._node:
            return
        t0 = time.monotonic()
        started = False
        while time.monotonic() - t0 < timeout:
            if not self._motion_done:
                started = True
            if started and self._motion_done:
                return
            time.sleep(0.02)

    def enable(self):
        while not self._srv_enable.wait_for_service(timeout_sec=2.0):
            self._node.get_logger().info('waiting for /enable_agx_arm ...')
        req = SetBool.Request()
        req.data = True
        self._srv_enable.call_async(req)

    def home(self):
        while not self._srv_home.wait_for_service(timeout_sec=2.0):
            self._node.get_logger().info('waiting for /move_home ...')
        self._srv_home.call_async(Empty.Request())

    def estop(self):
        # A fault/shutdown handler must never wait indefinitely for a dead driver.
        if self._node.context.ok() and self._srv_estop.service_is_ready():
            return self._srv_estop.call_async(Empty.Request())
        self._node.get_logger().error('Emergency stop service unavailable')

    def move_p(self, pose, wait=False, timeout=5.0):
        self._pub_move_p.publish(self._pose_msg(pose))
        if wait:
            self._wait(timeout)

    def move_p_quat(self, pose, wait=False, timeout=5.0):
        msg = PoseStamped()
        msg.header.frame_id = 'base_link'
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = pose[:3]
        msg.pose.orientation.x, msg.pose.orientation.y = pose[3], pose[4]
        msg.pose.orientation.z, msg.pose.orientation.w = pose[5], pose[6]
        self._pub_move_p.publish(msg)
        if wait:
            self._wait(timeout)

    def move_l(self, pose, wait=False, timeout=5.0):
        self._pub_move_l.publish(self._pose_msg(pose))
        if wait:
            self._wait(timeout)

    def move_j(self, joints, wait=False, timeout=5.0):
        msg = JointState()
        msg.name = [f'joint{i}' for i in range(1, len(joints) + 1)]
        msg.position = list(map(float, joints))
        self._pub_move_j.publish(msg)
        if wait:
            self._wait(timeout)

    def move_gripper(self, width_m, force=1.0, wait=False, timeout=3.0):
        msg = JointState()
        msg.name = ['gripper']
        msg.position = [float(width_m)]
        msg.effort = [float(force)]
        self._pub_joint.publish(msg)
        if wait:
            time.sleep(timeout if timeout < 1.0 else 1.0)

    def _pose_msg(self, pose):
        msg = PoseStamped()
        msg.header.frame_id = 'base_link'
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = pose[:3]
        q = self.rpy_to_quat(*pose[3:6])
        msg.pose.orientation.x, msg.pose.orientation.y = q[1], q[2]
        msg.pose.orientation.z, msg.pose.orientation.w = q[3], q[0]
        return msg


class SdkArmClient(ArmClient):
    """Direct pyAgxArm backend (mutually exclusive with agx_arm_ctrl)."""

    def __init__(self, channel='can0', gripper_max_range=0.07):
        from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, NeroFW
        cfg = create_agx_arm_config(
            robot=ArmModel.NERO,
            firmeware_version=NeroFW.DEFAULT,
            interface='socketcan',
            channel=channel,
        )
        self._robot = AgxArmFactory.create_arm(cfg)
        self._robot.connect()
        self._gripper = None
        self._gripper_max_range = gripper_max_range

    def enable(self):
        while not self._robot.enable():
            time.sleep(0.05)

    def home(self):
        self._robot.move_j([0.0] * self._robot.joint_nums)

    def estop(self):
        self._robot.electronic_emergency_stop()

    def move_p(self, pose, wait=False, timeout=5.0):
        self._robot.move_p(list(map(float, pose)))
        if wait:
            self._wait(timeout)

    def move_p_quat(self, pose, wait=False, timeout=5.0):
        from scipy.spatial.transform import Rotation as R
        q = [pose[3], pose[4], pose[5], pose[6]]  # xyzw
        r, p, y = R.from_quat(q).as_euler('xyz')
        self._robot.move_p([pose[0], pose[1], pose[2], r, p, y])
        if wait:
            self._wait(timeout)

    def move_l(self, pose, wait=False, timeout=5.0):
        self._robot.move_l(list(map(float, pose)))
        if wait:
            self._wait(timeout)

    def move_j(self, joints, wait=False, timeout=5.0):
        self._robot.move_j(list(map(float, joints)))
        if wait:
            self._wait(timeout)

    def move_gripper(self, width_m, force=1.0, wait=False, timeout=3.0):
        if self._gripper is None:
            self._gripper = self._robot.init_effector(
                self._robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
        self._gripper.move_gripper_m(min(width_m, self._gripper_max_range), force)
        if wait:
            time.sleep(0.3)

    def get_joint_state(self):
        return list(self._robot.get_joint_angles())

    def get_gripper_width(self):
        if self._gripper is None:
            return None
        st = self._gripper.get_gripper_status()
        return getattr(getattr(st, 'msg', None), 'width', None)

    def is_motion_done(self):
        st = self._robot.get_arm_status()
        return st is None or getattr(st.msg, 'motion_status', 0) == 0

    def _wait(self, timeout):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self.is_motion_done():
                return
            time.sleep(0.05)

    def set_limits(self, speed_percent=50):
        self._robot.set_speed_percent(speed_percent)
