"""Asynchronous, plan-only MoveIt RRTConnect client.

MoveIt validates self/environment collisions and returns a geometric joint
path.  Execution remains in the Nero controller so the mock ros2_control
bridge never competes with the direct AGX driver commands.
"""

import time

import numpy as np
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2


class RrtConnectPlanner:
    """Request a collision-checked arm path and expose it by polling."""

    def __init__(self, node, *, action_name='/move_action', group='arm',
                 planner_id='RRTConnectkConfigDefault', planning_time=2.0,
                 require_octomap=True, point_cloud_topic='/camera/depth/points'):
        try:
            from moveit_msgs.action import MoveGroup
            from moveit_msgs.srv import GetPlanningScene
        except ImportError as error:
            raise RuntimeError(
                'MoveIt messages are unavailable; source scripts/moveit_local_setup.bash') from error
        self.node = node
        self.MoveGroup = MoveGroup
        self.GetPlanningScene = GetPlanningScene
        self.action = ActionClient(node, MoveGroup, action_name)
        self.scene = node.create_client(GetPlanningScene, '/get_planning_scene')
        self.group = group
        self.planner_id = planner_id
        self.planning_time = float(planning_time)
        self.require_octomap = bool(require_octomap)
        self.cloud_time = 0.
        node.create_subscription(PointCloud2, point_cloud_topic,
                                 self._cloud_cb, qos_profile_sensor_data)
        self.phase = 'idle'
        self.future = None
        self.start = None
        self.target = None
        self.gripper_width = 0.
        self.started = 0.

    @property
    def busy(self):
        return self.phase != 'idle'

    def _cloud_cb(self, _message):
        self.cloud_time = time.monotonic()

    def cancel(self):
        if self.phase == 'result' and self.future is not None:
            # The result future does not expose a goal handle; planning is short
            # and plan-only, so dropping the result is sufficient on shutdown.
            self.future.cancel()
        self.phase = 'idle'
        self.future = None

    def begin(self, start_joints, target_joints, gripper_width=0.):
        if self.busy:
            raise ValueError('RRTConnect planner is already busy')
        self.start = np.asarray(start_joints, float)
        self.target = np.asarray(target_joints, float)
        if (self.start.shape != (7,) or self.target.shape != (7,)
                or not np.isfinite(np.r_[self.start, self.target]).all()):
            raise ValueError('RRTConnect requires two finite seven-joint states')
        self.gripper_width = float(gripper_width)
        if not 0. <= self.gripper_width <= .10:
            raise ValueError('RRTConnect gripper state must be within 0..0.10 m')
        if not self.action.server_is_ready():
            raise ValueError('MoveIt /move_action is unavailable; start the plan-only MoveIt launch')
        if self.require_octomap:
            if time.monotonic()-self.cloud_time > .6:
                raise ValueError(
                    'Camera point cloud is stale; RRTConnect obstacle planning is blocked')
            if not self.scene.service_is_ready():
                raise ValueError('MoveIt planning-scene service is unavailable')
            from moveit_msgs.msg import PlanningSceneComponents
            request = self.GetPlanningScene.Request()
            request.components.components = PlanningSceneComponents.OCTOMAP
            self.future = self.scene.call_async(request)
            self.phase = 'scene'
        else:
            self._send_goal()
        self.started = time.monotonic()

    def _send_goal(self):
        from moveit_msgs.msg import Constraints, JointConstraint
        goal = self.MoveGroup.Goal()
        request = goal.request
        request.group_name = self.group
        request.pipeline_id = 'ompl'
        request.planner_id = self.planner_id
        request.num_planning_attempts = 3
        request.allowed_planning_time = self.planning_time
        request.max_velocity_scaling_factor = .20
        request.max_acceleration_scaling_factor = .15
        request.workspace_parameters.header.frame_id = 'base_link'
        request.workspace_parameters.min_corner.x = -.70
        request.workspace_parameters.min_corner.y = -.70
        request.workspace_parameters.min_corner.z = -.25
        request.workspace_parameters.max_corner.x = .55
        request.workspace_parameters.max_corner.y = .55
        request.workspace_parameters.max_corner.z = .85
        arm_names = [f'joint{i}' for i in range(1, 8)]
        request.start_state.joint_state.name = arm_names+['gripper']
        request.start_state.joint_state.position = self.start.tolist()+[self.gripper_width]
        request.start_state.is_diff = False
        constraints = Constraints()
        constraints.name = 'fixed_bin_joint_goal'
        for name, position in zip(arm_names, self.target):
            constraint = JointConstraint()
            constraint.joint_name = name
            constraint.position = float(position)
            constraint.tolerance_above = .006
            constraint.tolerance_below = .006
            constraint.weight = 1.
            constraints.joint_constraints.append(constraint)
        request.goal_constraints = [constraints]
        goal.planning_options.plan_only = True
        goal.planning_options.look_around = False
        goal.planning_options.replan = False
        goal.planning_options.planning_scene_diff.is_diff = True
        self.future = self.action.send_goal_async(goal)
        self.phase = 'goal'

    def poll(self):
        """Return None while pending, or a list of seven-joint path points."""
        if self.phase == 'idle':
            raise ValueError('RRTConnect planner has no active request')
        if time.monotonic()-self.started > self.planning_time+8.:
            self.cancel()
            raise ValueError('RRTConnect planning timed out')
        if not self.future.done():
            return None
        if self.phase == 'scene':
            response = self.future.result()
            octomap = response.scene.world.octomap.octomap
            if self.require_octomap and not octomap.data:
                self.cancel()
                raise ValueError(
                    'MoveIt OctoMap is empty; camera point cloud cannot provide obstacle avoidance')
            self.node.get_logger().info(
                'RRTCONNECT_SCENE octomap_bytes=%d frame=%s' %
                (len(octomap.data), response.scene.world.octomap.header.frame_id))
            self._send_goal()
            return None
        if self.phase == 'goal':
            handle = self.future.result()
            if not handle.accepted:
                self.cancel()
                raise ValueError('MoveIt rejected the RRTConnect planning request')
            self.future = handle.get_result_async()
            self.phase = 'result'
            return None
        wrapped = self.future.result()
        result = wrapped.result
        if result.error_code.val != result.error_code.SUCCESS:
            code = result.error_code.val
            self.cancel()
            raise ValueError('RRTConnect failed with MoveIt error code %d' % code)
        trajectory = result.planned_trajectory.joint_trajectory
        expected = [f'joint{i}' for i in range(1, 8)]
        indices = []
        for name in expected:
            if name not in trajectory.joint_names:
                self.cancel()
                raise ValueError('RRTConnect trajectory is missing '+name)
            indices.append(trajectory.joint_names.index(name))
        path = [np.asarray([point.positions[index] for index in indices], float)
                for point in trajectory.points]
        if not path or np.max(np.abs(path[-1]-self.target)) > .02:
            self.cancel()
            raise ValueError('RRTConnect result does not end at the requested bin joint target')
        # The first point normally repeats the measured start state.  Do not
        # send a zero-length move to the AGX driver.
        while len(path) > 1 and np.max(np.abs(path[0]-self.start)) < .004:
            path.pop(0)
        self.node.get_logger().info(
            'RRTCONNECT_PLAN planner=%s planning_time=%.3f path_points=%d' %
            (self.planner_id, result.planning_time, len(path)))
        self.phase = 'idle'
        self.future = None
        return path
