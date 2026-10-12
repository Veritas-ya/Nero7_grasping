"""RGB optical geometry. All depths/points are metres; pixels are (u, v)."""
from dataclasses import dataclass
import cv2
import numpy as np


def depth_metres(image, encoding):
    if encoding == '16UC1' and image.dtype == np.uint16:
        depth = image.astype(np.float32) / 1000.
    elif encoding in ('32FC1', '64FC1') and image.dtype.kind == 'f':
        depth = image.astype(np.float32, copy=True)
    else:
        raise ValueError(f'Unsupported depth encoding/dtype: {encoding}/{image.dtype}')
    if depth.ndim != 2:
        raise ValueError('Depth must be single channel')
    depth[~np.isfinite(depth) | (depth <= 0)] = np.nan
    return depth


@dataclass
class CameraGeometry:
    K: np.ndarray
    width: int
    height: int
    frame_id: str
    D: tuple = ()
    distortion_model: str = 'plumb_bob'

    def __post_init__(self):
        self.K = np.asarray(self.K, float).reshape(3, 3)
        self.D = tuple(float(v) for v in self.D)
        if (not np.isfinite(self.K).all() or self.K[0, 0] <= 0 or self.K[1, 1] <= 0
                or not np.allclose(self.K[2], [0, 0, 1])
                or not np.allclose([self.K[0, 1], self.K[1, 0]], 0)
                or not np.isfinite(self.D).all() or self.width <= 0 or self.height <= 0
                or not self.frame_id):
            raise ValueError('Invalid camera intrinsics, dimensions or optical frame')
        if self.D and any(self.D) and self.distortion_model not in (
                'plumb_bob', 'rational_polynomial', 'equidistant'):
            raise ValueError('Unsupported distortion model: ' + self.distortion_model)
        if self.D and any(self.D):
            expected = (4,) if self.distortion_model == 'equidistant' else (4, 5, 8, 12, 14)
            if len(self.D) not in expected:
                raise ValueError('Invalid distortion coefficient count')

    @classmethod
    def from_info(cls, info, rectified=False):
        # Nontrivial ROI/binning need an explicit effective CameraInfo.
        if info.binning_x not in (0, 1) or info.binning_y not in (0, 1) or any(
                (info.roi.x_offset, info.roi.y_offset, info.roi.width, info.roi.height)):
            raise ValueError('CameraInfo ROI/binning requires an adjusted full-grid camera model')
        if rectified:
            rotation = np.asarray(info.r).reshape(3, 3)
            projection = np.asarray(info.p).reshape(3, 4)
            if not np.allclose(rotation, np.eye(3)) or not np.allclose(projection[:, 3], 0):
                raise ValueError('Nonidentity rectification/stereo projection is unsupported')
            return cls(projection[:, :3], info.width, info.height, info.header.frame_id)
        return cls(info.k, info.width, info.height, info.header.frame_id,
                   tuple(info.d), info.distortion_model)

    def check_grid(self, image, frame_id):
        if image.shape[:2] != (self.height, self.width) or frame_id != self.frame_id:
            raise ValueError('Image grid/frame does not match color CameraInfo; no resize is allowed')

    def rays(self, u, v):
        pixels = np.column_stack([np.asarray(u).ravel(), np.asarray(v).ravel()]).astype(float)
        if self.D and any(self.D):
            function = cv2.fisheye.undistortPoints if self.distortion_model == 'equidistant' else cv2.undistortPoints
            xy = function(pixels.reshape(-1, 1, 2), self.K, np.asarray(self.D)).reshape(-1, 2)
        else:
            xy = (pixels - self.K[:2, 2]) / [self.K[0, 0], self.K[1, 1]]
        return np.column_stack([xy, np.ones(len(xy))])

    def project(self, points):
        points = np.asarray(points, float).reshape(-1, 3)
        if self.D and any(self.D):
            if self.distortion_model == 'equidistant':
                pixels, _ = cv2.fisheye.projectPoints(points.reshape(-1, 1, 3), np.zeros(3),
                                                       np.zeros(3), self.K, np.asarray(self.D))
            else:
                pixels, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3), self.K, np.asarray(self.D))
            return pixels.reshape(-1, 2)
        return points[:, :2] / points[:, 2:3] * [self.K[0, 0], self.K[1, 1]] + self.K[:2, 2]

    def cloud(self, depth, transform):
        self.check_grid(depth, self.frame_id)
        transform = check_transform(transform)
        points = self.grid_rays() * depth[..., None]
        return points @ transform[:3, :3].T + transform[:3, 3]

    def grid_rays(self):
        if not hasattr(self, '_ray_grid'):
            v, u = np.indices((self.height, self.width))
            self._ray_grid = self.rays(u, v).reshape(self.height, self.width, 3)
        return self._ray_grid


def check_transform(transform):
    transform = np.asarray(transform, float)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1])
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1., atol=1e-5)):
        raise ValueError('Invalid optical-to-base rigid transform')
    return transform


def roi_mask(shape, roi):
    h, w = shape[:2]
    if roi is None or not any(roi):
        return np.ones((h, w), bool)
    u0, v0, u1, v1 = map(int, roi)
    if not (0 <= u0 < u1 <= w and 0 <= v0 < v1 <= h):
        raise ValueError('ROI must lie inside the color image grid')
    mask = np.zeros((h, w), bool)
    mask[v0:v1, u0:u1] = True
    return mask
