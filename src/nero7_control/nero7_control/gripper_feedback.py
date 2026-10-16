"""Confirm a gripper command from fresh, distinct device feedback samples."""
import math


class GripperFeedbackMonitor:
    def __init__(self, max_age=.3, tolerance=.003):
        self.max_age = max_age
        self.tolerance = tolerance
        self.sample = None
        self.command_time = None
        self.baseline_stamp = float('-inf')
        self.goal = None
        self.confirmed_stamps = []

    def update(self, width, stamp, received, enabled, fault=False):
        self.sample = (width, stamp, received, enabled, fault)

    def begin(self, goal, now):
        self.goal = goal
        self.command_time = now
        self.baseline_stamp = self.sample[1] if self.sample else float('-inf')
        self.confirmed_stamps = []

    def reading(self, now):
        if self.sample is None or self.command_time is None:
            return None
        width, stamp, received, enabled, fault = self.sample
        if not (math.isfinite(width) and 0 <= width <= .105 and math.isfinite(stamp)):
            raise ValueError('Invalid gripper feedback width or timestamp')
        if not 0 <= now-received <= self.max_age:
            return None
        if received < self.command_time or stamp <= self.baseline_stamp:
            return None
        if fault:
            raise ValueError('Gripper driver reports a fault')
        if not enabled:
            return None
        return width

    def opening_confirmed(self, now):
        width = self.reading(now)
        if width is None or width < self.goal-self.tolerance:
            self.confirmed_stamps = []
            return False
        stamp = self.sample[1]
        if self.confirmed_stamps and stamp < self.confirmed_stamps[-1]:
            self.confirmed_stamps = []
            return False
        if not self.confirmed_stamps or stamp > self.confirmed_stamps[-1]:
            self.confirmed_stamps.append(stamp)
        return (len(self.confirmed_stamps) >= 3
                and self.confirmed_stamps[-1]-self.confirmed_stamps[0] >= .1)
