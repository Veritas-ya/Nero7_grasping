"""Lead-time sweep calibration (M3).

Sweeps the FSM lead parameter k_lead and measures the true timing error:
delta_s = (predicted block position - gate position) at the grasp moment,
published by grasp_controller on /grasp/delta_s.  The best k_lead minimizes
|mean(delta_s)|.

Usage (system running, belt at the target speed, feed blocks one by one):
    ros2 run nero7_control lead_calib --ros-args \
        -p k_leads:="[0.0, 0.05, 0.1, 0.15]" -p trials:=5

The tool sets k_lead on the running controller (ros2 param), collects
`trials` delta_s samples per value, prints the table and writes the best
value into controller.yaml.
"""
import json
import os

import numpy as np
import rclpy
from rclpy.node import Node
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue
from std_msgs.msg import Float64
import yaml


class LeadCalibNode(Node):
    def __init__(self):
        super().__init__('lead_calib')
        k_leads = self.declare_parameter('k_leads', '[0.0, 0.05, 0.1, 0.15]').value
        self._k_leads = json.loads(k_leads) if isinstance(k_leads, str) else k_leads
        self._trials = self.declare_parameter('trials', 5).value
        self._yaml_out = self.declare_parameter(
            'yaml_out',
            '/home/ubuntu22/Nero7_grasping/src/nero7_control/config/controller.yaml').value
        self._param_client = self.create_client(
            SetParameters, '/grasp_controller/set_parameters')
        self._samples = []
        self.create_subscription(Float64, '/grasp/delta_s', self._delta_cb, 5)
        self._timer = self.create_timer(1.0, self._step)
        self._idx = 0
        self.get_logger().info(f'sweeping k_lead over {self._k_leads}')

    def _delta_cb(self, msg):
        self._samples.append(msg.data)
        self.get_logger().info(f'[k={self._k_leads[self._idx]:.3f}] delta_s '
                               f'= {msg.data * 1000:.1f} mm '
                               f'({len(self._samples)}/{self._trials})')

    def _set_k(self, v):
        req = SetParameters.Request()
        p = Parameter()
        p.name = 'k_lead'
        p.value = ParameterValue(type=3, double_value=float(v))  # PARAMETER_DOUBLE
        req.parameters = [p]
        fut = self._param_client.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=2.0)

    def _step(self):
        if self._idx >= len(self._k_leads):
            return
        if self._param_client.wait_for_service(timeout_sec=1.0):
            self._timer.cancel()
            self._run_sweep()
            raise SystemExit(0)

    def _run_sweep(self):
        results = {}
        for k in self._k_leads:
            self.get_logger().info(f'>>> k_lead = {k} (feed blocks now)')
            self._set_k(k)
            self._samples = []
            while len(self._samples) < self._trials:
                rclpy.spin_once(self, timeout_sec=0.2)
            arr = np.array(self._samples)
            results[k] = arr
            self.get_logger().info(f'>>> k={k}: mean {arr.mean() * 1000:.1f} mm, '
                                   f'|mean| {abs(arr.mean()) * 1000:.1f} mm, '
                                   f'std {arr.std() * 1000:.1f} mm')
        best = min(results, key=lambda k: abs(results[k].mean()))
        self.get_logger().info(f'=== best k_lead = {best} ===')
        # write into controller.yaml
        try:
            with open(self._yaml_out) as f:
                data = yaml.safe_load(f)
            data['grasp_controller']['ros__parameters']['k_lead'] = float(best)
            with open(self._yaml_out, 'w') as f:
                yaml.safe_dump(data, f)
            self.get_logger().info(f'written k_lead={best} to {self._yaml_out}')
        except OSError as e:
            self.get_logger().warn(f'could not update yaml: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = LeadCalibNode()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
