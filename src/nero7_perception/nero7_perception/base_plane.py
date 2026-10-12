"""Empty-belt calibration in base coordinates, independent of ROS/control."""
from dataclasses import dataclass
from pathlib import Path
import time
import numpy as np
import yaml

from .geometry import check_transform


@dataclass
class BaseBeltPlane:
    normal: np.ndarray
    offset: float
    mode: str
    diagnostics: dict

    def __post_init__(self):
        normal = np.asarray(self.normal, float)
        if (normal.shape != (3,) or not np.isfinite(normal).all()
                or not np.isfinite(self.offset) or np.linalg.norm(normal) <= 0
                or self.mode not in ('fixed', 'ransac3d')):
            raise ValueError('Invalid base plane')
        length = np.linalg.norm(normal)
        self.normal, self.offset = normal / length, float(self.offset / length)
        if self.normal[2] < 0:
            self.normal, self.offset = -self.normal, -self.offset
        if self.normal[2] < .5:
            raise ValueError('Belt plane is too steep to define vertical height')
        if self.mode == 'fixed' and not np.allclose(self.normal, [0, 0, 1]):
            raise ValueError('Fixed plane must have base +Z normal')

    def signed_distance(self, points):
        return np.einsum('...i,i->...', points, self.normal) + self.offset

    def z_at(self, x, y):
        return -(self.normal[0] * np.asarray(x) + self.normal[1] * np.asarray(y) + self.offset) / self.normal[2]

    def vertical_height(self, points):
        return self.signed_distance(points) / self.normal[2]

    def height_image(self, depth, camera, transform):
        # Algebraically identical to transforming every XYZ point, without an HxWx3 allocation.
        camera.check_grid(depth, camera.frame_id)
        transform = check_transform(transform)
        key = (id(camera), transform.tobytes(), self.normal.tobytes())
        if getattr(self, '_height_cache_key', None) != key:
            self._height_coefficient = np.einsum('...i,i->...', camera.grid_rays(),
                                                transform[:3, :3].T @ self.normal)
            self._height_cache_key = key
        coefficient = self._height_coefficient
        return (depth * coefficient + self.normal @ transform[:3, 3] + self.offset) / self.normal[2]


def fit_ransac3d(points, distance=.006, min_ratio=.6, max_iter=100, max_points=5000, seed=1):
    """Three noncollinear XYZ samples; unit-normal orthogonal metric residual."""
    points = np.asarray(points, float).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 100:
        raise ValueError('At least 100 finite belt points are required')
    rng = np.random.default_rng(seed)
    if len(points) > max_points:
        points = points[rng.choice(len(points), max_points, replace=False)]
    best, count = None, 0
    for _ in range(max_iter):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-10:
            continue
        normal /= length
        mask = np.abs(np.einsum('ij,j->i', points - a, normal)) < distance
        if mask.sum() > count:
            best, count = mask, int(mask.sum())
    if best is None or count < min_ratio * len(points):
        raise ValueError('No dominant 3D plane; check empty ROI and height prior')
    # Refine and reselect inliers twice, rather than retaining the sampled triplet.
    for _ in range(2):
        center = np.mean(points[best], axis=0)
        _, singular, vt = np.linalg.svd(points[best] - center, full_matrices=False)
        if singular[1] < 1e-5:
            raise ValueError('Belt samples have no two-dimensional spatial support')
        normal = vt[-1]
        if normal[2] < 0:
            normal = -normal
        offset = -float(normal @ center)
        best = np.abs(np.einsum('ij,j->i', points, normal) + offset) < distance
        if best.sum() < max(100, min_ratio * len(points)):
            raise ValueError('Refined plane has insufficient inliers')
    residual = np.einsum('ij,j->i', points[best], normal) + offset
    if normal[2] < .5:
        raise ValueError('Dominant plane is too steep to be the belt')
    diagnostics = dict(sample_count=len(points), inlier_count=int(best.sum()),
                       inlier_ratio=float(best.mean()), residual_std_m=float(np.std(residual)),
                       residual_p95_m=float(np.percentile(np.abs(residual), 95)),
                       tilt_rad=float(np.arccos(np.clip(normal[2], -1, 1))),
                       slope_x=float(-normal[0] / normal[2]), slope_y=float(-normal[1] / normal[2]),
                       xy_span_m=np.ptp(points[best, :2], axis=0).tolist())
    return BaseBeltPlane(normal, offset, 'ransac3d', diagnostics), points[best]


class EmptyBeltCalibration:
    def __init__(self, mode='fixed', frames=15, distance=.006, min_ratio=.6,
                 max_tilt=.01, max_flat_residual=.006, max_height_spread=.004,
                 min_xy_span=.05, max_points=5000, max_3d_tilt=.2):
        if mode not in ('fixed', 'ransac3d') or frames < 3:
            raise ValueError('Calibration needs fixed/ransac3d mode and >=3 frames')
        self.mode, self.frames = mode, frames
        self.distance, self.min_ratio = distance, min_ratio
        self.max_tilt, self.max_flat_residual = max_tilt, max_flat_residual
        self.max_height_spread, self.min_xy_span = max_height_spread, min_xy_span
        self.max_points = max_points
        self.max_3d_tilt = max_3d_tilt
        self.samples, self.stamps = [], []

    def add(self, points, stamp):
        if not np.isfinite(stamp) or (self.stamps and stamp <= self.stamps[-1]):
            return False
        points = np.asarray(points, float).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) < 100:
            raise ValueError('Too few valid green belt points')
        if len(points) > self.max_points:
            rng = np.random.default_rng(len(self.samples))
            points = points[rng.choice(len(points), self.max_points, replace=False)]
        self.samples.append(points)
        self.stamps.append(float(stamp))
        return len(self.samples) >= self.frames

    def fit(self):
        if len(self.samples) < self.frames:
            raise ValueError('Insufficient distinct empty-belt frames')
        plane, inliers = fit_ransac3d(np.concatenate(self.samples), self.distance,
                                     self.min_ratio, max_points=self.max_points)
        if plane.diagnostics['tilt_rad'] > self.max_3d_tilt:
            raise ValueError('Dominant plane exceeds the permitted 3D belt tilt; inspect ROI/TF')
        if min(plane.diagnostics['xy_span_m']) < self.min_xy_span:
            raise ValueError('ROI is too small in base X/Y to assess parallelism')
        heights = []
        for points in self.samples:
            accepted = np.abs(plane.signed_distance(points)) < self.distance
            if accepted.mean() < self.min_ratio:
                raise ValueError('A calibration frame is inconsistent with the belt plane')
            # Residual medians compare the same plane, unaffected by spatial sampling.
            heights.append(float(np.median(plane.signed_distance(points[accepted])) / plane.normal[2]))
        spread = float(np.ptp(heights))
        h = float(np.median(inliers[:, 2]))
        flat = inliers[:, 2] - h
        diagnostics = dict(plane.diagnostics, frame_count=len(self.samples),
                           frame_height_spread_m=spread, median_base_z_m=h,
                           fixed_height_std_m=float(np.std(flat)),
                           fixed_height_p95_m=float(np.percentile(np.abs(flat), 95)))
        # Expose spatial Z statistics so tilt and TF rotation errors can be inspected.
        diagnostics['height_by_xy_bins'] = []
        edges = [np.linspace(inliers[:, axis].min(), inliers[:, axis].max()+1e-9, 4) for axis in (0, 1)]
        for ix in range(3):
            for iy in range(3):
                mask = ((inliers[:, 0] >= edges[0][ix]) & (inliers[:, 0] < edges[0][ix+1])
                        & (inliers[:, 1] >= edges[1][iy]) & (inliers[:, 1] < edges[1][iy+1]))
                if mask.sum() >= 10:
                    diagnostics['height_by_xy_bins'].append(dict(count=int(mask.sum()),
                        xy_m=np.median(inliers[mask, :2], axis=0).tolist(),
                        median_z_m=float(np.median(inliers[mask, 2])), std_z_m=float(np.std(inliers[mask, 2]))))
        if spread > self.max_height_spread:
            raise ValueError('Belt height is unstable across empty frames')
        if self.mode == 'fixed':
            if diagnostics['tilt_rad'] > self.max_tilt or diagnostics['fixed_height_p95_m'] > self.max_flat_residual:
                raise ValueError(f'Horizontal assumption failed: tilt={diagnostics["tilt_rad"]:.5f} rad, '
                                 f'p95={diagnostics["fixed_height_p95_m"]:.5f} m; inspect TF or use ransac3d')
            return BaseBeltPlane(np.array([0., 0., 1.]), -h, 'fixed', diagnostics)
        return BaseBeltPlane(plane.normal, plane.offset, 'ransac3d', diagnostics)


def plane_context(camera, transform, base_frame, d2c_offset=(0, 0)):
    return dict(base_frame=base_frame, camera_frame=camera.frame_id,
                K=camera.K.tolist(), D=list(camera.D), distortion_model=camera.distortion_model,
                image_size=[camera.width, camera.height], optical_to_base=check_transform(transform).tolist(),
                d2c_offset=list(d2c_offset))


def save_plane(path, plane, camera, transform, base_frame, roi, d2c_offset=(0, 0)):
    data = dict(plane_context(camera, transform, base_frame, d2c_offset),
                schema_version=1, mode=plane.mode, normal=plane.normal.tolist(), offset=plane.offset,
                diagnostics=plane.diagnostics, plane_roi_px=list(roi), units='m')
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + '.tmp')
    temporary.write_text(yaml.safe_dump(data, sort_keys=False))
    temporary.replace(target)


def archive_plane(path):
    """A failed recalibration must not leave an old file presented as the new result."""
    target = Path(path)
    if not target.exists():
        return None
    previous = target.with_name(f'{target.stem}.previous.{time.time_ns()}{target.suffix}')
    target.rename(previous)
    return str(previous)


def load_plane(path):
    data = yaml.safe_load(Path(path).read_text())
    if data.get('schema_version') != 1 or data.get('units') != 'm':
        raise ValueError('Unsupported plane calibration schema/units')
    return BaseBeltPlane(data['normal'], data['offset'], data['mode'], data['diagnostics']), data


def check_plane_context(data, camera, transform, base_frame, d2c_offset=(0, 0)):
    reference = check_transform(data['optical_to_base'])
    transform = check_transform(transform)
    rotation_delta = reference[:3, :3].T @ transform[:3, :3]
    angle = np.arccos(np.clip((np.trace(rotation_delta) - 1) / 2, -1, 1))
    if (data['base_frame'] != base_frame or data['camera_frame'] != camera.frame_id
            or data['image_size'] != [camera.width, camera.height]
            or not np.allclose(data['K'], camera.K, atol=1e-6, rtol=0)
            or len(data['D']) != len(camera.D) or not np.allclose(data['D'], camera.D, atol=1e-8, rtol=0)
            or data['distortion_model'] != camera.distortion_model
            or data['d2c_offset'] != list(d2c_offset)
            or np.linalg.norm(reference[:3, 3] - transform[:3, 3]) > .002 or angle > .005):
        raise ValueError('Plane calibration camera/TF context changed; recalibration is required')
