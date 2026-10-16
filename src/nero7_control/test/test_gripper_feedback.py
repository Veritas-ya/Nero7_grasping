import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nero7_control.gripper_feedback import GripperFeedbackMonitor
from nero7_control.static_once import StaticOnce


def feed(monitor, width=.070, stamp=100., received=10., enabled=True, fault=False):
    monitor.update(width, stamp, received, enabled, fault)


def test_repeated_closed_feedback_never_confirms_opening():
    monitor = GripperFeedbackMonitor()
    monitor.begin(.070, 10.)
    for i in range(60):
        at = 10.+i*.1
        feed(monitor, .003273, 100.+i*.1, at)
        assert not monitor.opening_confirmed(at)


def test_opening_needs_three_distinct_new_frames_over_time():
    monitor = GripperFeedbackMonitor()
    feed(monitor)
    monitor.begin(.070, 10.)
    assert not monitor.opening_confirmed(10.01)
    for at in (10.02, 10.04, 10.06):
        feed(monitor, stamp=100.01, received=at)
        assert not monitor.opening_confirmed(at)
    feed(monitor, stamp=100.07, received=10.07)
    assert not monitor.opening_confirmed(10.07)
    feed(monitor, stamp=100.15, received=10.15)
    assert monitor.opening_confirmed(10.15)
    assert not monitor.opening_confirmed(10.6)


@pytest.mark.parametrize('width', [float('nan'), float('inf'), -.001, .2])
def test_invalid_width_is_rejected(width):
    monitor = GripperFeedbackMonitor()
    monitor.begin(.070, 10.)
    feed(monitor, width=width)
    with pytest.raises(ValueError):
        monitor.reading(10.)


def test_disabled_or_old_feedback_cannot_confirm():
    monitor = GripperFeedbackMonitor()
    feed(monitor)
    monitor.begin(.070, 10.1)
    feed(monitor, stamp=100.1, received=10.2, enabled=False)
    assert monitor.reading(10.2) is None
    feed(monitor, stamp=100.2, received=10.2)
    assert monitor.reading(10.6) is None
    feed(monitor, stamp=100., received=10.7)
    assert monitor.reading(10.7) is None


def test_fault_is_reported():
    monitor = GripperFeedbackMonitor()
    monitor.begin(.070, 10.)
    feed(monitor, fault=True)
    with pytest.raises(ValueError, match='fault'):
        monitor.opening_confirmed(10.)


def opening_actor(elapsed):
    actor = StaticOnce.__new__(StaticOnce)
    actor.state, actor.started = 'STATIC_OPEN', 0.
    actor.open_width = .070
    actor.motion_issued = True
    actor.arm, actor.node = Mock(), Mock()
    actor.node.get_clock().now.return_value = SimpleNamespace(nanoseconds=100_000_000_000)
    actor.status = SimpleNamespace(err_status=0, arm_status=0)
    actor._fresh = lambda: True
    actor._motion_time = lambda: elapsed
    actor._motion = Mock()
    actor.plan = dict(above=[0., 0., 0.], joints=[[0.]*7])
    profile_path = Path(__file__).parents[1]/'config/fixed_bin.yaml'
    actor.bin_profile = dict(
        profile_path=str(profile_path),
        sha256=hashlib.sha256(profile_path.read_bytes()).hexdigest())
    actor.gripper_feedback = GripperFeedbackMonitor()
    actor.gripper_feedback.begin(.070, 0.)
    actor.arm.get_gripper_width.return_value = .003273
    return actor


def test_open_phase_does_not_approach_after_fixed_delay():
    actor = opening_actor(1.6)
    actor.tick()
    actor._motion.assert_not_called()
    assert actor.state == 'STATIC_OPEN'


def test_open_timeout_stops_without_approach():
    actor = opening_actor(4.1)
    actor.tick()
    actor._motion.assert_not_called()
    actor.arm.estop.assert_called_once()
    assert actor.state == 'STATIC_ABORT'
    assert 'opening not confirmed' in actor.failure


@pytest.mark.parametrize('state', ['STATIC_RELEASE', 'STATIC_RECOVER_OPEN'])
def test_release_does_not_continue_on_unconfirmed_opening(state):
    actor = opening_actor(1.6)
    actor.state, actor.failure = state, None
    actor.tick()
    actor._motion.assert_not_called()
    assert actor.state == state


def test_fresh_confirmed_opening_allows_approach(monkeypatch):
    actor = opening_actor(.4)
    actor.gripper_feedback.opening_confirmed = Mock(return_value=True)
    actor.tick()
    actor._motion.assert_called_once_with(actor.plan['above'], 'STATIC_APPROACH', actor.plan['joints'][0])


def test_closing_without_feedback_does_not_lift_or_recover():
    actor = opening_actor(1.6)
    actor.state = 'STATIC_CLOSE'
    actor._grip = Mock()
    actor.tick()
    actor._motion.assert_not_called()
    actor._grip.assert_not_called()
    assert actor.state == 'STATIC_CLOSE'
