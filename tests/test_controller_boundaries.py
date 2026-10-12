"""Execute production controller methods with explicit mocked ROS/driver boundaries.

Only class definitions are loaded; imports requiring a robot workspace are
replaced by test dependencies. No ROS executor, CAN driver or GPU is exercised.
"""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock
import math
import time
import numpy as np
import pytest

from nero7_control.fsm import GraspFsm
from nero7_control.belt import BeltModel

ROOT = Path(__file__).resolve().parents[1]


def production_class(path, name, dependencies):
    tree = ast.parse((ROOT/path).read_text())
    definition = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    module = ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[]))
    namespace = dict(dependencies)
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace[name]


@pytest.fixture
def controller():
    cls = production_class('src/nero7_control/nero7_control/grasp_controller.py',
                           'GraspControllerNode', dict(Node=object, HALT='HALT', math=math, np=np))
    node = cls.__new__(cls)
    node._static_once = None
    node._p = dict(dry_run=False, feedback_timeout=.2)
    node._continuous_motion_issued = True
    node._watchdog_tripped = False
    node._fsm, node._arm = GraspFsm(BeltModel(), {}), Mock()
    node._fsm._enter('WAIT')
    node._fsm._enter = Mock(wraps=node._fsm._enter)
    node._fsm.tick = Mock(wraps=node._fsm.tick)
    node._arm.estop.return_value = None
    node.get_parameter = Mock(return_value=NS(value=False))
    node.get_logger = Mock(return_value=Mock())
    return node


def test_disabling_active_continuous_execution_requests_stop_once(controller):
    controller._tick()
    controller._tick()
    controller._arm.estop.assert_called_once()
    assert controller._watchdog_tripped and controller._fsm.state == 'HALT'
    controller._fsm._enter.assert_called_once_with('HALT')
    controller._arm.move_l.assert_not_called()
    controller._arm.move_gripper.assert_not_called()


def test_preview_does_not_stop_an_arm_it_has_not_commanded(controller):
    controller._continuous_motion_issued = False
    controller._tick()
    controller._arm.estop.assert_not_called()
    controller._fsm.tick.assert_not_called()


class ClockValue:
    def __init__(self, seconds):
        self.nanoseconds = int(seconds*1e9)

    def __sub__(self, other):
        return NS(nanoseconds=self.nanoseconds-other.nanoseconds)


def test_feedback_timeout_stops_accepted_motion(controller):
    controller.get_parameter.return_value.value = True
    controller._feedback_time = ClockValue(0.)
    controller.get_clock = Mock(return_value=NS(now=lambda: ClockValue(1.)))
    controller._tick()
    controller._arm.estop.assert_called_once()
    controller._fsm.tick.assert_not_called()


def test_invalid_pose_stops_and_does_not_send_recovery_motion(controller):
    controller._pose_ok = Mock(return_value=False)
    controller._execute(dict(type='move_l', pose=[0.]*6))
    controller._arm.estop.assert_called_once()
    controller._arm.move_l.assert_not_called()
    controller._arm.home.assert_not_called()


@pytest.mark.parametrize('enabled,dry_run', [(False, False), (True, True)])
def test_static_disable_or_dry_run_stops_active_cycle(enabled, dry_run):
    cls = production_class('src/nero7_control/nero7_control/static_once.py',
                           'StaticOnce', dict(time=time, np=np))
    actor = cls.__new__(cls)
    actor.state, actor.motion_issued = 'STATIC_APPROACH', True
    actor.node, actor.arm = Mock(), Mock()
    actor.node._p = dict(dry_run=dry_run)
    actor.node.get_parameter.return_value.value = enabled
    actor._motion_time = lambda: 0.
    actor.tick()
    assert actor.state == 'STATIC_ABORT'
    actor.arm.estop.assert_called_once()
    actor.arm.move_gripper.assert_not_called()


def test_old_idle_status_does_not_complete_a_new_motion():
    cls = production_class('src/nero7_control/nero7_control/arm_client.py',
                           'RosTopicArmClient', dict(ArmClient=object, Node=object, time=time))
    arm = cls.__new__(cls)
    arm._motion_done = True
    arm._motion_commanded()
    arm._status_cb(NS(motion_status=0))
    assert not arm.is_motion_done()
    arm._status_cb(NS(motion_status=1))
    assert not arm.is_motion_done()
    arm._status_cb(NS(motion_status=0))
    assert arm.is_motion_done()


def test_startup_waits_for_first_feedback_without_latching_a_fault(controller):
    controller.get_parameter.return_value.value = True
    controller._continuous_motion_issued = False
    controller._feedback_time = ClockValue(0.)
    controller.get_clock = Mock(return_value=NS(now=lambda: ClockValue(1.)))
    controller._tick()
    assert not controller._watchdog_tripped
    controller._fsm._enter.assert_not_called()
    controller._arm.estop.assert_not_called()
