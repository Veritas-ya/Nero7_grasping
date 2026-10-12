"""Explicit pytest plugin: mock the GPU model, retain real ROS node/types."""
import sys
import types


class InferenceDouble:
    def __init__(self, *args, **kwargs):
        self.requests = []

    def request(self, *args, **kwargs):
        self.requests.append((args, kwargs))

    def latest(self, max_age=.2):
        return None

    def shutdown(self):
        pass


module = types.ModuleType('nero7_control.grasp_inference')
module.GraspInference = InferenceDouble
sys.modules['nero7_control.grasp_inference'] = module
