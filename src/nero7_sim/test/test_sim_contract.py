"""Hardware geometry, depth units, and simulation time contracts."""
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock
import xml.etree.ElementTree as ET
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from nero7_sim.scene import robot_overlay
from nero7_sim.driver import cartesian_path
from nero7_sim.camera_sync import CameraSync
from nero7_control.kinematics import Kinematics
from nero7_control.static_once import StaticOnce
from nero7_control.grasp_controller import GraspControllerNode

ROOT = Path(__file__).resolve().parents[3]
URDF = '/home/ubuntu22/nero_ws/install/nero_description/share/nero_description/urdf/nero_description.urdf'


def test_sim_overlay_preserves_every_hardware_joint_transform(tmp_path):
    configuration = yaml.safe_load((ROOT/'src/nero7_bringup/config/calibration.yaml').read_text())
    overlay = robot_overlay(URDF, [0.]*7, configuration['gripper_tcp'])
    path = tmp_path/'robot.urdf'; ET.ElementTree(overlay).write(path)
    hardware, simulation = Kinematics(URDF), Kinematics(path)
    for q in [np.zeros(7), [.1, -.2, .3, .4, -.5, .2, -.3]]:
        np.testing.assert_allclose(simulation.fk(q), hardware.fk(q), atol=1e-12)
    np.testing.assert_allclose(simulation.lower, hardware.lower)
    np.testing.assert_allclose(simulation.upper, hardware.upper)


def test_gazebo_metres_are_normalized_to_network_millimetres():
    node = GraspControllerNode.__new__(GraspControllerNode)
    node._bridge = Mock()
    node._bridge.imgmsg_to_cv2.return_value = np.array([[.92, np.nan, np.inf, -1.]], dtype=np.float32)
    node._depth_cb(NS(encoding='32FC1', header=NS(stamp='sample')))
    np.testing.assert_allclose(node._depth, [[920., 0., 0., 0.]])
    assert node._depth_stamp == 'sample'


def test_hardware_uint16_depth_is_preserved():
    node = GraspControllerNode.__new__(GraspControllerNode); node._bridge = Mock()
    image = np.array([[920, 1100]], dtype=np.uint16)
    node._bridge.imgmsg_to_cv2.return_value = image
    node._depth_cb(NS(encoding='16UC1', header=NS(stamp='sample')))
    assert node._depth is image


def test_arrival_settling_uses_simulation_seconds():
    actor = StaticOnce.__new__(StaticOnce)
    actor.node = Mock(); actor.use_sim_time = True
    actor.started = 0.; actor.settled = None
    actor._actual = lambda: (np.zeros(3), Rotation.identity())
    actor.goal = (np.zeros(3), Rotation.identity()); actor.status = NS(motion_status=0)
    actor.node.get_clock.return_value.now.return_value.nanoseconds = 400_000_000
    assert not actor._arrived()
    actor.node.get_clock.return_value.now.return_value.nanoseconds = 750_000_000
    assert actor._arrived()


def test_cartesian_path_respects_linear_speed_and_inverse_kinematics():
    kin = Kinematics(URDF)
    q = np.array([-.4449193, .276879, -.4254938, 2.0089263, 2.5207441, .5516288, -.2842443])
    original = kin.fk(q); goal = original[:3, 3]+[0., 0., -.04]
    times, path = cartesian_path(kin, q, goal, Rotation.from_matrix(original[:3, :3]), .028, .6)
    assert times[-1] >= .04/.028-1e-8
    np.testing.assert_allclose(kin.fk(path[-1])[:3, 3], goal, atol=1e-4)
    for t, joints in zip(times, path):
        expected = original[:3, 3]+[0., 0., -.04*t/times[-1]]
        np.testing.assert_allclose(kin.fk(joints)[:3, 3], expected, atol=1e-4)


def test_upstream_object_and_grasp_line_are_inside_example_camera_roi():
    calibration = yaml.safe_load((ROOT/'src/nero7_bringup/config/calibration.yaml').read_text())
    belt = yaml.safe_load((ROOT/'src/nero7_bringup/config/belt.yaml').read_text())
    example = yaml.safe_load((ROOT/'src/nero7_sim/config/example.yaml').read_text())
    transform = np.array(calibration['camera_to_base_cv'])
    direction = np.array(belt['belt_direction'])
    gate = np.array(belt['belt_origin'])+direction*belt['gate_s']
    u0, v0, u1, v1 = example['roi_px']
    for distance in [0., example['spawn_distance_upstream']]:
        world = np.r_[gate-direction*distance, belt['belt_height']+example['object_size'][2]]
        optical = transform[:3, :3].T@(world-transform[:3, 3])
        uv = np.array([example['camera_width']/2, example['camera_height']/2])+example['camera_fx']*optical[:2]/optical[2]
        assert u0+20 < uv[0] < u1-20 and v0+15 < uv[1] < v1-15


def test_rendered_rgbd_waits_for_clock_without_changing_measurement():
    sync = CameraSync.__new__(CameraSync)
    sync.frames, sync.last_stamp = {}, -1
    sync.image_publishers = {kind: Mock() for kind in ('color', 'depth')}
    now = NS(nanoseconds=1_000_000_000)
    sync.get_clock = lambda: NS(now=lambda: now)
    color = NS(header=NS(stamp=NS(sec=1, nanosec=0)))
    depth = NS(header=NS(stamp=NS(sec=1, nanosec=0)))
    sync.receive('color', color); sync.receive('depth', depth)
    sync.flush()
    sync.image_publishers['color'].publish.assert_not_called()
    now.nanoseconds += 40_000_000
    sync.flush()
    sync.image_publishers['color'].publish.assert_called_once_with(color)
    sync.image_publishers['depth'].publish.assert_called_once_with(depth)
    assert color.header.stamp.sec == 1 and color.header.stamp.nanosec == 0


def test_missing_matching_depth_does_not_publish_rgb_only_or_stale_frame():
    sync = CameraSync.__new__(CameraSync)
    sync.frames, sync.last_stamp = {}, -1
    sync.image_publishers = {kind: Mock() for kind in ('color', 'depth')}
    sync.get_clock = lambda: NS(now=lambda: NS(nanoseconds=1_200_000_000))
    sync.receive('color', NS(header=NS(stamp=NS(sec=1, nanosec=0))))
    sync.receive('depth', NS(header=NS(stamp=NS(sec=1, nanosec=100_000_000))))
    sync.flush()
    for publisher in sync.image_publishers.values(): publisher.publish.assert_not_called()
    assert 1_000_000_000 not in sync.frames
