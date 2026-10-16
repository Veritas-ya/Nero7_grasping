from unittest.mock import Mock

import numpy as np
import pytest

pytest.importorskip('moveit_msgs')

from nero7_control.moveit_rrtconnect import RrtConnectPlanner


def test_goal_explicitly_requests_plan_only_rrtconnect_with_gripper_state():
    planner = RrtConnectPlanner.__new__(RrtConnectPlanner)
    from moveit_msgs.action import MoveGroup
    planner.MoveGroup = MoveGroup
    planner.group = 'arm'
    planner.planner_id = 'RRTConnectkConfigDefault'
    planner.planning_time = 2.
    planner.start = np.arange(7, dtype=float)*.01
    planner.target = np.arange(7, dtype=float)*-.02
    planner.gripper_width = .043
    planner.action = Mock()

    planner._send_goal()

    goal = planner.action.send_goal_async.call_args.args[0]
    assert goal.request.pipeline_id == 'ompl'
    assert goal.request.planner_id == 'RRTConnectkConfigDefault'
    assert goal.request.group_name == 'arm'
    assert goal.planning_options.plan_only is True
    assert goal.request.start_state.joint_state.name[-1] == 'gripper'
    assert goal.request.start_state.joint_state.position[-1] == pytest.approx(.043)
    assert [c.joint_name for c in goal.request.goal_constraints[0].joint_constraints] == [
        f'joint{i}' for i in range(1, 8)]
