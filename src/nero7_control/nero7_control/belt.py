"""Belt model for the control package.

Mirrors ``nero7_perception.conveyor_model.BeltModel`` so the controller can
run without the perception package loaded (e.g. M0 arm-only tests).
"""
import numpy as np


class BeltModel:
    """Belt direction / gate position in base frame, loaded from belt.yaml."""

    def __init__(self, d_hat=(1.0, 0.0), p0=(0.0, 0.0), gate_s=0.35):
        self.d_hat = np.asarray(d_hat, float)
        n = np.linalg.norm(self.d_hat)
        if n > 0:
            self.d_hat = self.d_hat / n
        self.p0 = np.asarray(p0, float)
        self.gate_s = float(gate_s)

    @classmethod
    def from_yaml(cls, data):
        d = data.get('belt_direction', [1.0, 0.0])
        p0 = data.get('belt_origin', [0.0, 0.0])
        gate_s = data.get('gate_s', 0.35)
        return cls(d, p0, gate_s)

    def s_of(self, point_base):
        """Scalar position along belt direction."""
        p = np.asarray(point_base, float)[:2] - self.p0
        return float(np.dot(p, self.d_hat))

    def point_of(self, s):
        """Base-frame xy point at scalar position s."""
        return self.p0 + self.d_hat * s
