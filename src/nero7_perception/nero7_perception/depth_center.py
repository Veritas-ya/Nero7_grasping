"""Approximate side-grasp centers from aligned depth, for a planar conveyor.

The center is the midpoint of a visible footprint rectangle and a dominant
surface height above the belt. It is not a measured center of mass.
"""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def short_side_orientations(grasp_rpy, closing_axis, direction_xy, horizontal=False):
    """Yaw alternatives aligning the measured flange closing axis with short edge.

    closing_axis must describe the mounted gripper in flange coordinates;
    axis direction signs are equivalent for a symmetric two-finger gripper.
    Preserve the calibrated approach tilt, and recompute TCP for each result.
    """
    rotation = Rotation.from_euler('xyz', grasp_rpy)
    axis = np.asarray(closing_axis, float)
    direction = np.asarray(direction_xy, float)
    if axis.shape != (3,) or direction.shape != (2,) or not np.isfinite(axis).all() or not np.isfinite(direction).all():
        raise ValueError('Invalid closing axis or footprint direction')
    if np.linalg.norm(axis) < 1e-6 or np.linalg.norm(direction) < 1e-6:
        raise ValueError('Closing axis and direction must be nonzero')
    axis /= np.linalg.norm(axis)
    current = rotation.apply(axis)
    if np.linalg.norm(current[:2]) < .5:
        raise ValueError('Closing axis must be approximately horizontal')
    angle = np.arctan2(direction[1], direction[0])-np.arctan2(current[1], current[0])
    angle = (angle+np.pi/2) % np.pi-np.pi/2
    options = []
    for yaw in [angle, angle+np.pi if angle < 0 else angle-np.pi]:
        result = Rotation.from_euler('z', yaw)*rotation
        if horizontal:
            tilted = result.apply(axis)
            level = np.r_[tilted[:2]/np.linalg.norm(tilted[:2]), 0.0]
            correction, _ = Rotation.align_vectors([level], [tilted])
            result = correction*result
        options.append(dict(yaw_change=float(yaw),
                            rpy=result.as_euler('xyz').tolist(),
                            closing_axis_base=result.apply(axis).tolist()))
    return options


def estimate_depth_centers(depth_m, intrinsics, camera_to_base, roi,
                           min_area=300, clearance=.012, max_opening=.070,
                           min_surface_height=.008):
    depth = np.asarray(depth_m, float)
    K = np.asarray(intrinsics, float)
    T = np.asarray(camera_to_base, float)
    if depth.ndim != 2 or K.shape != (3, 3) or T.shape != (4, 4):
        raise ValueError('Expected depth HxW, intrinsics 3x3 and extrinsics 4x4')
    if not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0:
        raise ValueError('Invalid intrinsics')
    if not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1]):
        raise ValueError('Invalid extrinsics')
    u0, v0, u1, v1 = map(int, roi)
    h, w = depth.shape
    if not (0 <= u0 < u1 <= w and 0 <= v0 < v1 <= h):
        raise ValueError('ROI must lie inside depth image')
    vv, uu = np.mgrid[v0:v1, u0:u1]
    z = depth[v0:v1, u0:u1]
    valid = np.isfinite(z) & (z > .1) & (z < 3)
    cam = np.stack([(uu-K[0, 2])*z/K[0, 0],
                    (vv-K[1, 2])*z/K[1, 1], z], axis=-1)
    points = cam @ T[:3, :3].T + T[:3, 3]
    cloud = points[valid]
    if len(cloud) < 1000:
        raise ValueError('Too few valid depth pixels to fit belt')
    rng = np.random.default_rng(7)
    sample = cloud[rng.choice(len(cloud), min(len(cloud), 10000), replace=False)]
    A = np.c_[sample[:, :2], np.ones(len(sample))]
    best = None
    best_count = 0
    for _ in range(80):
        idx = rng.choice(len(sample), 3, replace=False)
        if np.linalg.cond(A[idx]) > 1e6:
            continue
        plane = np.linalg.solve(A[idx], sample[idx, 2])
        if np.linalg.norm(plane[:2]) > .3:
            continue  # near-horizontal belt in base coordinates
        inliers = np.abs(A @ plane - sample[:, 2]) < .004
        count = int(inliers.sum())
        if count > best_count:
            best_count, best = count, inliers
    if best is None or best_count < len(sample)*.55:
        raise ValueError('No dominant near-horizontal belt plane')
    plane = np.linalg.lstsq(A[best], sample[best, 2], rcond=None)[0]
    belt_z = points[..., 0]*plane[0] + points[..., 1]*plane[1] + plane[2]
    height = points[..., 2] - belt_z
    if not .008 <= min_surface_height < .10:
        raise ValueError('Surface height threshold outside supported range')
    mask = (valid & (height >= min_surface_height) & (height <= .12)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    plans = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        member = labels == label
        obj = points[member]
        heights = height[member]
        # A dominant surface band avoids choosing a small tall protrusion.
        bins = np.arange(.008, .125, .003)
        hist, edges = np.histogram(heights, bins)
        dominant = int(np.argmax(hist))
        band = (heights >= edges[dominant]) & (heights <= edges[dominant+1])
        body_height = float(np.median(heights[band]))
        xy = obj[:, :2]
        low, high = np.percentile(xy, [1, 99], axis=0)
        trimmed = xy[np.all((xy >= low) & (xy <= high), axis=1)]
        if len(trimmed) < 30:
            continue
        rect = cv2.minAreaRect(trimmed.astype(np.float32))
        center_xy = np.asarray(rect[0], float)
        box = cv2.boxPoints(rect).astype(float)
        vectors = np.roll(box, -1, axis=0)-box
        lengths = np.linalg.norm(vectors, axis=1)
        short = int(np.argmin(lengths))
        width = float(lengths[short])
        direction = vectors[short]/max(width, 1e-9)
        local_belt = float(np.r_[center_xy, 1] @ plane)
        center_z = local_belt + body_height/2
        target_z = max(center_z, local_belt + clearance,
                       local_belt + min_surface_height + .002)
        opening = width + .010
        reason = None
        if opening > max_opening:
            reason = 'estimated width exceeds gripper opening'
        elif body_height <= clearance:
            reason = 'body too low for fingertip clearance'
        elif target_z >= local_belt + body_height - .003:
            reason = 'insufficient side contact height'
        plans.append(dict(center_base=[*center_xy.tolist(), float(center_z)],
                          fingertip_target_base=[*center_xy.tolist(), float(target_z)],
                          belt_z=local_belt, body_height=body_height,
                          width=width, opening=opening,
                          length=float(np.max(lengths)),
                          min_surface_height=float(min_surface_height),
                          closing_direction_xy=direction.tolist(),
                          area_px=area, feasible=reason is None, rejection=reason))
    return dict(belt_plane_z_from_xy=plane.tolist(),
                belt_inlier_ratio=best_count/len(sample), candidates=plans)
