"""Actual ROS JointState serialization/discovery; no hardware driver is run."""
import time
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import JointState
from nero7_control.arm_client import RosTopicArmClient
from nero7_control.grasp_controller import GraspControllerNode
from ament_index_python.packages import get_package_share_directory
from tests.test_ros_startup import minimal_urdf


@pytest.fixture(autouse=True)
def ros_context():
    rclpy.init(args=[])
    yield
    rclpy.shutdown()


def spin_until(executor, predicate, timeout=3.):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=.02)
        if predicate():
            return
    assert predicate(), 'ROS fixture discovery/message timeout'


def test_move_js_message_contract_and_exclusive_endpoint():
    sender, receiver, competitor = Node('follow_sender'), Node('follow_driver_double'), Node('competing_sender')
    executor = SingleThreadedExecutor()
    received = []
    for node in [sender, receiver, competitor]:
        executor.add_node(node)
    arm = RosTopicArmClient(sender)
    try:
        assert not arm.can_stream_joints()
        with pytest.raises(RuntimeError):
            arm.stream_joints([0.]*7)
        receiver.create_subscription(JointState, 'control/move_js', received.append, 1)
        spin_until(executor, arm.can_stream_joints)
        arm.stream_joints([.01*i for i in range(7)])
        spin_until(executor, lambda: bool(received))
        assert received[0].name == ['joint%d' % i for i in range(1, 8)]
        assert received[0].position == pytest.approx([.01*i for i in range(7)])
        assert received[0].header.stamp.sec > 0
        assert list(received[0].velocity) == []  # AGX move_js uses position references.
        competitor.create_publisher(JointState, 'control/move_js', 1)
        spin_until(executor, lambda: sender.count_publishers('/control/move_js') == 2)
        assert not arm.can_stream_joints()
        with pytest.raises(RuntimeError):
            arm.stream_joints([0.]*7)
    finally:
        for node in [sender, receiver, competitor]:
            executor.remove_node(node)
            node.destroy_node()
        executor.shutdown()


@pytest.mark.parametrize('joints', [[0.]*6, [0.]*8, [float('nan')]+[0.]*6])
def test_invalid_joint_stream_never_serializes_a_command(joints):
    node = Node('invalid_stream')
    try:
        arm = RosTopicArmClient(node)
        with pytest.raises(ValueError):
            arm.stream_joints(joints)
    finally:
        node.destroy_node()


def test_installed_follow_config_and_explicit_overrides_start_in_preview(tmp_path):
    urdf = tmp_path/'robot.urdf'
    minimal_urdf(urdf)
    config = Path(get_package_share_directory('nero7_control'))/'config/follow.yaml'
    assert config.is_file()
    initializer = Node.__init__
    overrides = [Parameter('mode', value='conveyor_follow'),
        Parameter('params_file', value=str(config)), Parameter('urdf_path', value=str(urdf)),
        Parameter('visualization_enabled', value=False), Parameter('follow_command_latency', value=.04)]
    def initialize(instance, *args, **kwargs):
        return initializer(instance, *args, parameter_overrides=overrides, **kwargs)
    with patch.object(Node, '__init__', initialize):
        node = GraspControllerNode()
    try:
        actor = node._static_once
        assert actor.state == 'FOLLOW_WAIT_TARGET'
        assert not actor._execution_enabled()
        assert actor.latency == .04
        assert node._p['infer_tf_timeout'] == 0.
        node._arm.stream_joints = Mock()
        node._arm.move_gripper = Mock()
        node._arm.move_j = Mock()
        actor.tick()
        assert not actor.motion_issued
        node._arm.stream_joints.assert_not_called()
        node._arm.move_gripper.assert_not_called()
        node._arm.move_j.assert_not_called()
    finally:
        node._static_once.shutdown()
        node.destroy_node()
