"""Conveyor belt model: plane fitting, belt-direction scalar coordinates and
speed estimation.

The belt plane is fitted in depth space (z = a*u + b*v + c) with RANSAC, which
is valid for a near-top-down camera.  The belt direction and gate position come
from ``belt.yaml`` (produced by ``belt_calib``).  Speed estimation ports the
sliding-window linear fit with residual gating from the conveyor_vision sim
(``speed_measure.py``).
"""
import numpy as np


class BeltPlane:
    """RANSAC plane fit in (u, v, z) depth space."""

    def __init__(self, distance=0.006, min_ratio=0.4, max_iter=100, rng_seed=1, max_points=5000):
        if max_points < 100 or max_iter < 1:
            raise ValueError('Plane fitting needs >=100 samples and >=1 iteration')
        self.max_points = int(max_points)
        self.distance = distance
        self.min_ratio = min_ratio
        self.max_iter = max_iter
        self.rng = np.random.default_rng(rng_seed)
        self.coeffs = None  # [a, b, c] for z = a*u + b*v + c
        self.last_good = None

    def fit(self, depth, roi=None):
        """Fit plane to depth (HxW float32 m, NaN invalid).

        ``roi``: (u0, v0, u1, v1) restrict fit region (belt area).
        Returns True on success.
        """
        d = depth
        if roi is not None:
            u0, v0, u1, v1 = roi
            d = d[v0:v1, u0:u1]
            off_u, off_v = u0, v0
        else:
            off_u, off_v = 0, 0
        h, w = d.shape
        vv, uu = np.mgrid[0:h, 0:w]
        z = d
        valid = np.isfinite(z) & (z > 0)
        us = (uu + off_u)[valid].astype(np.float64)
        vs = (vv + off_v)[valid].astype(np.float64)
        zs = z[valid].astype(np.float64)
        if zs.size < 100:
            self.coeffs = self.last_good
            return self.coeffs is not None

        # Bound work, and avoid threaded BLAS for a tiny three-column product.
        if zs.size > self.max_points:
            idx = self.rng.choice(zs.size, self.max_points, replace=False)
            us, vs, zs = us[idx], vs[idx], zs[idx]

        A = np.column_stack([us, vs, np.ones_like(us)])
        best = None
        best_inliers = 0
        required_iterations = self.max_iter
        best_mask = None
        for iteration in range(self.max_iter):
            idx = self.rng.choice(zs.size, 3, replace=False)
            if abs(np.linalg.det(A[idx])) < 1e-8:
                continue
            try:
                coeffs, *_ = np.linalg.lstsq(A[idx], zs[idx], rcond=None)
            except np.linalg.LinAlgError:
                continue
            resid = np.abs(us*coeffs[0]+vs*coeffs[1]+coeffs[2]-zs)
            mask = resid < self.distance
            n_in = int(mask.sum())
            if n_in > best_inliers:
                best_inliers = n_in
                best = coeffs
                best_mask = mask
                success = min((n_in/zs.size)**3, 1.-1e-12)
                required_iterations = max(10, int(np.ceil(np.log(.001)/np.log1p(-success))))
            if iteration+1 >= required_iterations:
                break
        if best is None or best_inliers < self.min_ratio * zs.size:
            self.coeffs = self.last_good
            return self.coeffs is not None
        for _ in range(2):
            best, *_ = np.linalg.lstsq(A[best_mask], zs[best_mask], rcond=None)
            best_mask = np.abs(us*best[0]+vs*best[1]+best[2]-zs) < self.distance
            if best_mask.sum() < max(100, self.min_ratio*zs.size):
                self.coeffs = self.last_good
                return self.coeffs is not None
        self.coeffs = best
        self.last_good = best
        return True

    def plane_z(self, u, v):
        """Belt surface z (m) at pixel(s) (u, v)."""
        if self.coeffs is None:
            return np.nan
        a, b, c = self.coeffs
        return a * np.asarray(u, float) + b * np.asarray(v, float) + c


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


class SpeedEstimator:
    """Sliding-window scalar speed fit along the belt direction."""

    def __init__(self, window_s=0.6, min_samples=8, min_span=0.3,
                 resid_std_max=0.008, jump_clear=0.1):
        self.window_s = window_s
        self.min_samples = min_samples
        self.min_span = min_span
        self.resid_std_max = resid_std_max
        self.jump_clear = jump_clear
        self.window = []  # (t, s)
        self.v = None
        self.ok = False

    def update(self, t, s):
        """Add sample; returns (v, ok).  v is None until the fit converges."""
        if not (np.isfinite(t) and np.isfinite(s)):
            self.clear()
            return self.v, self.ok
        if self.window and t == self.window[-1][0]:
            # Reusing the same image cannot provide another velocity sample.
            return self.v, self.ok
        if self.window and t < self.window[-1][0]:
            self.clear()
        if self.window and abs(s - self.window[-1][1]) > self.jump_clear:
            self.clear()
        self.window.append((t, s))
        cutoff = t - self.window_s
        while self.window and self.window[0][0] < cutoff:
            self.window.pop(0)
        # A gap or a short current window invalidates the previous fit.
        self.v = None
        self.ok = False
        if (len(self.window) >= self.min_samples and
                self.window[-1][0] - self.window[0][0] >= self.min_span):
            ts = np.array([w[0] for w in self.window])
            ss = np.array([w[1] for w in self.window])
            # ROS image stamps are Unix seconds; center before fitting.
            ts -= ts[0]
            a, b = np.polyfit(ts, ss, 1)
            resid = float(np.std(ss - (a * ts + b)))
            self.v = float(a)
            self.ok = resid <= self.resid_std_max
        return self.v, self.ok

    def clear(self):
        self.window.clear()
        self.v = None
        self.ok = False
