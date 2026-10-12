"""Stopping a controller must retain ROS long enough to issue the stop request."""
from concurrent.futures import Future
from unittest.mock import Mock, patch

from nero7_control.arm_client import RosTopicArmClient
from nero7_control.static_once import StaticOnce
from nero7_control import grasp_controller


def test_missing_stop_service_returns_without_waiting():
    arm = RosTopicArmClient.__new__(RosTopicArmClient)
    arm._node = Mock()
    arm._srv_estop = Mock()
    arm._srv_estop.service_is_ready.return_value = False
    assert arm.estop() is None
    arm._srv_estop.wait_for_service.assert_not_called()
    arm._srv_estop.call_async.assert_not_called()


def test_active_shutdown_waits_bounded_time_for_stop_acknowledgement():
    actor = StaticOnce.__new__(StaticOnce)
    actor.node, actor.arm, actor.motion_issued = Mock(), Mock(), True
    future = Future()
    actor.arm.estop.return_value = future
    with patch('nero7_control.static_once.rclpy.spin_until_future_complete') as spin:
        actor.shutdown()
    actor.arm.estop.assert_called_once()
    assert spin.call_args.kwargs['timeout_sec'] == .5


def test_preview_shutdown_does_not_stop_or_command_arm():
    actor = StaticOnce.__new__(StaticOnce)
    actor.arm, actor.motion_issued = Mock(), False
    actor.shutdown()
    actor.arm.estop.assert_not_called()


def test_interrupt_cancels_tick_and_stops_before_ros_shutdown():
    node = Mock()
    events = []
    node._timer.cancel.side_effect = lambda: events.append('timer_cancel')
    node._static_once.shutdown.side_effect = lambda: events.append('arm_stop')
    with patch.object(grasp_controller, 'GraspControllerNode', return_value=node), \
            patch.object(grasp_controller, 'signal'), patch.object(grasp_controller, 'rclpy') as ros:
        ros.spin.side_effect = KeyboardInterrupt
        ros.try_shutdown.side_effect = lambda: events.append('ros_shutdown')
        grasp_controller.main()
    assert events == ['timer_cancel', 'arm_stop', 'ros_shutdown']
    node.destroy_node.assert_called_once()
