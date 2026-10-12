"""Real Humble node startup and launch resolution, with driver/network doubles."""
import importlib.util
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import Image, CameraInfo
from builtin_interfaces.msg import Time
from nero7_msgs.msg import BlockTrackArray
from nero7_control.grasp_controller import GraspControllerNode
from nero7_perception.block_detector import BlockDetectorNode
from ament_index_python.packages import get_package_share_directory

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def ros_context():
    rclpy.init(args=[])
    yield
    rclpy.shutdown()


def minimal_urdf(path):
    joints = ''.join(
        '<joint name="joint%d" type="revolute"><origin xyz="0 0 .05" rpy="0 0 0"/>'
        '<axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14"/></joint>' % i
        for i in range(1, 8))
    path.write_text('<robot name="test">'+joints+'</robot>')


@pytest.mark.parametrize('mode', ['conveyor', 'static_once', 'conveyor_once'])
def test_formal_modes_start_without_parameter_collisions_or_commands(tmp_path, mode):
    urdf = tmp_path/'test.urdf'
    minimal_urdf(urdf)
    node_init = Node.__init__
    overrides = [Parameter('mode', value=mode), Parameter('execution_enabled', value=False),
                 Parameter('visualization_enabled', value=False)]
    if mode != 'conveyor':
        overrides.append(Parameter('urdf_path', value=str(urdf)))

    def initialize(instance, *args, **kwargs):
        return node_init(instance, *args, parameter_overrides=overrides, **kwargs)

    with patch.object(Node, '__init__', initialize):
        node = GraspControllerNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin_once(timeout_sec=.05)
        assert node.get_parameter('execution_enabled').value is False
        assert node.get_parameter('dry_run').value is False
        assert Path(node.get_parameter('calibration_file').value).is_file()
        assert not node._continuous_motion_issued
        if node._static_once is not None:
            assert not node._static_once.motion_issued
    finally:
        executor.remove_node(node)
        if node._static_once is not None:
            node._static_once.shutdown()
        node.destroy_node()
        executor.shutdown()


def test_detector_constructs_and_diagnostics_runs_with_actual_ros_types():
    node = BlockDetectorNode()
    try:
        node._status_cb()
        assert node._p['speed_window_s'] == 1.2
        assert node._p['debug_rate_hz'] == 10.
        node._process()  # Missing image data must not publish a fake empty belt.
        assert node._diagnostics['processed_frames'] == 0
    finally:
        node.destroy_node()


@pytest.mark.parametrize('filename', ['bringup.launch.py', 'dry_run.launch.py'])
def test_installed_launch_configuration_paths_and_overrides(filename):
    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument
    path = Path(get_package_share_directory('nero7_bringup'))/'launch'/filename
    spec = importlib.util.spec_from_file_location('test_launch', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    context = LaunchContext()
    for entity in module.generate_launch_description().entities:
        if isinstance(entity, DeclareLaunchArgument):
            context.launch_configurations[entity.name] = ''.join(
                item.perform(context) for item in entity.default_value)
    assert Path(context.launch_configurations['perception_params']).is_file()
    assert Path(context.launch_configurations['controller_params']).is_file()
    assert context.launch_configurations['execution_enabled'] == 'false'


def test_registered_rgbd_controller_check_uses_real_message_metadata():
    node_init = Node.__init__
    def initialize(instance, *args, **kwargs):
        return node_init(instance, *args, parameter_overrides=[
            Parameter('visualization_enabled', value=False)], **kwargs)
    with patch.object(Node, '__init__', initialize):
        node = GraspControllerNode()
    try:
        color = CameraInfo(width=640, height=480)
        color.header.frame_id = 'camera_color_optical_frame'
        color.k = [500., 0., 320., 0., 500., 240., 0., 0., 1.]
        color.d = [0.]*5
        color.distortion_model = 'plumb_bob'
        depth = CameraInfo(width=640, height=480)
        depth.header.frame_id = color.header.frame_id
        depth.k, depth.d, depth.distortion_model = color.k, color.d, color.distortion_model
        node._info_cb(color)
        node._depth_info_cb(depth)
        node._color, node._depth = np.zeros((480, 640, 3), np.uint8), np.ones((480, 640), np.uint16)
        node._color_frame = node._depth_frame = color.header.frame_id
        assert node._registered_images_valid()
        depth.header.frame_id = 'camera_depth_optical_frame'
        assert not node._registered_images_valid()
    finally:
        node.destroy_node()
