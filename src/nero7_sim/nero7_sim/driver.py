"""Gazebo adapter exposing the same motion/feedback API as the AGX driver.

Joint and Cartesian commands are interpolated at configured speed limits.
Actual Gazebo joint/link feedback supplies flange pose and the blocked jaw width.
This is an ideal position actuator example; it does not emulate CAN/motor dynamics.
"""
from concurrent.futures import ThreadPoolExecutor
import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import SetBool, Empty
from agx_arm_msgs.msg import AgxArmStatus, GripperStatus
from scipy.spatial.transform import Rotation, Slerp
from nero7_control.kinematics import Kinematics


def cartesian_path(kin, seed, position, rotation, linear_speed, joint_speed):
    start = kin.fk(seed)
    distance = np.linalg.norm(position-start[:3, 3])
    angle = (Rotation.from_matrix(start[:3, :3]).inv()*rotation).magnitude()
    count = max(2, int(math.ceil(max(distance/.008, angle/.04))))
    fractions = np.linspace(0., 1., count+1)
    orientations = Slerp([0., 1.], Rotation.concatenate([Rotation.from_matrix(start[:3, :3]), rotation]))
    path = [np.array(seed, float)]
    for alpha in fractions[1:]:
        point = start[:3, 3]*(1-alpha)+position*alpha
        path.append(kin.solve(point, orientations(alpha), path[-1]))
    path = np.array(path)
    duration = max(.3, distance/linear_speed, angle/joint_speed,
                   np.max(np.abs(np.diff(path, axis=0)))*count/joint_speed)
    return fractions*duration, path


class SimDriver(Node):
    def __init__(self):
        super().__init__('driver')
        declare = lambda name, default: self.declare_parameter(name, default).value
        self.kin = Kinematics(declare('urdf_path', ''))
        self.speed = int(declare('speed_percent', 20))
        self.linear_speed = float(declare('cartesian_speed_at_100_percent', .14))*self.speed/100
        self.joint_speed = float(declare('joint_speed_at_100_percent', 3.))*self.speed/100
        self.gripper_speed = float(declare('gripper_speed', .20))
        self.q = np.asarray(declare('initial_joints', [0.]*7), float)
        self.home_q = self.q.copy()
        self.enabled, self.fault = False, 0
        self.actual_q, self.actual_width, self.raw_received = None, .07, 0.
        self.actual_pose = None
        self.width, self.width_goal = .07, .07
        self.trajectory, self.planning = None, None
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.last_time = self.seconds()
        self.command = self.create_publisher(Float64MultiArray, 'gazebo/joint_commands', 10)
        self.feedback = self.create_publisher(JointState, 'feedback/joint_states', 10)
        self.pose_pub = self.create_publisher(PoseStamped, 'feedback/tcp_pose', 10)
        self.status_pub = self.create_publisher(AgxArmStatus, 'feedback/arm_status', 10)
        self.grip_pub = self.create_publisher(GripperStatus, 'feedback/gripper_status', 10)
        self.create_subscription(JointState, 'gazebo/joint_states', self.raw, 10)
        self.create_subscription(PoseStamped, 'gazebo/flange_pose', self.raw_pose, 10)
        self.create_subscription(JointState, 'control/move_j', self.move_j, 10)
        self.create_subscription(PoseStamped, 'control/move_l', self.move_l, 10)
        self.create_subscription(PoseStamped, 'control/move_p', self.move_l, 10)
        self.create_subscription(JointState, 'control/joint_states', self.grip, 10)
        self.create_service(SetBool, 'enable_agx_arm', self.enable)
        self.create_service(Empty, 'emergency_stop', self.stop)
        self.create_service(Empty, 'move_home', self.home)
        self.timer = self.create_timer(.02, self.tick)

    def seconds(self):
        return self.get_clock().now().nanoseconds*1e-9

    def raw(self, msg):
        mapping = dict(zip(msg.name, msg.position))
        if all(f'joint{i}' in mapping for i in range(1, 8)):
            self.actual_q = np.array([mapping[f'joint{i}'] for i in range(1, 8)])
            self.actual_width = 2*(mapping['left_finger_joint']-.003)
            self.raw_received = time.monotonic()

    def raw_pose(self, msg):
        self.actual_pose = msg

    def enable(self, request, response):
        self.enabled = bool(request.data)
        if self.enabled: self.fault = 0
        else: self.stop(request, response)
        response.success = True
        return response

    def home(self, request, response):
        self.move_j(JointState(position=self.home_q.tolist()))
        return response

    def stop(self, request, response):
        self.enabled = False; self.trajectory = None
        if self.planning is not None: self.planning.cancel()
        self.planning = None
        if self.actual_q is not None: self.q = self.actual_q.copy()
        self.width_goal = self.width = self.actual_width
        return response

    def move_j(self, msg):
        if not self.enabled: return
        goal = np.array(msg.position[:7], float)
        if goal.shape != (7,) or not np.isfinite(goal).all() or np.any(goal<self.kin.lower) or np.any(goal>self.kin.upper):
            self.fault = 1; return
        seed = self.actual_q.copy() if self.actual_q is not None else self.q.copy()
        duration = max(.3, np.max(np.abs(goal-seed))/self.joint_speed)
        self.trajectory = (self.seconds(), np.array([0., duration]), np.array([seed, goal]))

    def move_l(self, msg):
        if not self.enabled or self.planning is not None: return
        try:
            p, q = msg.pose.position, msg.pose.orientation
            position = np.array([p.x, p.y, p.z])
            rotation = Rotation.from_quat([q.x, q.y, q.z, q.w])
            if not np.isfinite(position).all(): raise ValueError('Invalid position')
            seed = self.actual_q.copy() if self.actual_q is not None else self.q.copy()
            self.planning = self.pool.submit(cartesian_path, self.kin, seed, position, rotation,
                                            self.linear_speed, self.joint_speed)
        except Exception as error:
            self.fault = 1; self.get_logger().error(str(error))

    def grip(self, msg):
        if self.enabled and msg.position and np.isfinite(msg.position[0]):
            self.width_goal = float(np.clip(msg.position[0], 0., .07))

    def tick(self):
        now = self.seconds(); dt = max(0., min(now-self.last_time, .1)); self.last_time = now
        if self.planning is not None and self.planning.done():
            future, self.planning = self.planning, None
            try:
                times, positions = future.result()
                if self.enabled: self.trajectory = (now, times, positions)
            except Exception as error:
                self.fault = 1; self.get_logger().error('Cartesian preflight failed: '+str(error))
        moving = self.planning is not None
        if self.trajectory is not None:
            started, times, positions = self.trajectory
            elapsed = now-started
            self.q = np.array([np.interp(elapsed, times, positions[:, i]) for i in range(7)])
            moving = elapsed < times[-1]
            if not moving: self.trajectory = None
        self.width += float(np.clip(self.width_goal-self.width, -self.gripper_speed*dt, self.gripper_speed*dt))
        self.command.publish(Float64MultiArray(data=[*self.q.tolist(), self.width]))
        # Feedback must come from Gazebo, never from the requested trajectory.
        if self.actual_q is None or time.monotonic()-self.raw_received > .2:
            return
        joints = JointState(); joints.header.stamp = self.get_clock().now().to_msg()
        joints.name = [f'joint{i}' for i in range(1, 8)]; joints.position = self.actual_q.tolist()
        self.feedback.publish(joints)
        if self.actual_pose is not None:
            self.pose_pub.publish(self.actual_pose)
        self.status_pub.publish(AgxArmStatus(arm_status=0, motion_status=int(moving), err_status=self.fault))
        gripper = GripperStatus(); gripper.header = joints.header; gripper.width = self.actual_width
        gripper.driver_enable_status = self.enabled; self.grip_pub.publish(gripper)


def main(args=None):
    rclpy.init(args=args); node = SimDriver()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.pool.shutdown(wait=True, cancel_futures=True)
        node.destroy_node(); rclpy.try_shutdown()
