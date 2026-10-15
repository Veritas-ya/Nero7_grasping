"""Switchable RGB/depth detection, feeding the unchanged BlockTracker contract."""
import math
import cv2
import numpy as np
import yaml
from pathlib import Path

from .geometry import roi_mask, check_transform


DETECTION_DEFAULTS = {
    'detection_mode': 'legacy_depth',  # Explicit baseline; new modes need empty-belt calibration.
    'plane_mode': 'fixed',
    'plane_file': '',
    'plane_calibration_enabled': False,
    'plane_roi_px': [0, 0, 0, 0],
    'plane_calibration_frames': 15,
    'plane_calibration_max_points': 5000,
    'base_plane_min_ratio': .7,
    'plane_max_tilt_rad': .01,
    'plane_max_3d_tilt_rad': .2,
    'plane_max_flat_residual_m': .006,
    'plane_max_height_spread_m': .004,
    'plane_min_xy_span_m': .05,
    'plane_height_hint_enabled': False,
    'plane_height_hint': 0.,
    'plane_height_hint_tolerance_m': .03,
    'plane_empty_gray_max_ratio': .005,
    'sync_max_delta_s': .02,
    'image_max_age_s': .2,
    'sync_buffer_size': 8,
    'color_rectified': False,
    'apply_d2c_offset': False,
    'legacy_rgb_components': False,
    'legacy_min_edge_support': .15,
    'lower_green': [35, 40, 40],
    'upper_green': [85, 255, 255],
    'gray_saturation_max': 70,
    'gray_value_min': 35,
    'gray_value_max': 245,
    'depth_min_valid_pixels': 30,
    'depth_frame_min_valid_ratio': .1,
    'depth_min_valid_ratio': .5,
    'depth_erode_px': 1,
    'depth_region_mad_scale': 3.,
    'depth_region_min_band_m': .003,
    'rgb_height_min_m': 0.,
    'rgb_height_max_m': .12,
}


def resolve_detection_paths(params, params_path=None):
    """Resolve default belt/calibration resources in source or isolated installs."""
    params = dict(params)
    source_root = Path(params_path).resolve().parent.parent.parent if params_path else None
    for name, filename in [('belt_file', 'belt.yaml'), ('calibration_file', 'calibration.yaml')]:
        value = params.get(name, '')
        if value:
            target = Path(value).expanduser()
            if not target.is_absolute() and params_path:
                target = Path(params_path).resolve().parent/target
            params[name] = str(target)
            continue
        candidate = source_root/'nero7_bringup/config'/filename if source_root else None
        if candidate is not None and candidate.is_file():
            params[name] = str(candidate)
        else:
            from ament_index_python.packages import get_package_share_directory
            params[name] = str(Path(get_package_share_directory('nero7_bringup'))/'config'/filename)
    return params


def load_detection_params(path):
    params = dict(DETECTION_DEFAULTS, roi_px=[0, 0, 0, 0], block_height_min=.008,
                  block_height_max=.060, min_area_px=200, plane_ransac_distance=.006,
                  plane_ransac_min_ratio=.4, plane_refresh_every=10,
                  workspace_x=[-.7, .7], workspace_y=[-.7, .7], workspace_z=[-.25, .15])
    with open(path) as stream:
        params.update(yaml.safe_load(stream)['block_detector']['ros__parameters'])
    validate_detection_config(params)
    return resolve_detection_paths(params, path)


def validate_detection_config(params):
    if not np.isfinite(params.get('legacy_min_edge_support', .15)) or not 0 <= params.get('legacy_min_edge_support', .15) <= 1:
        raise ValueError('legacy_min_edge_support must lie in [0,1]')
    if params['detection_mode'] not in ('legacy_depth', 'depth_plane', 'rgb', 'rgb_depth'):
        raise ValueError('detection_mode must be legacy_depth/depth_plane/rgb/rgb_depth')
    if params['plane_mode'] not in ('fixed', 'ransac3d'):
        raise ValueError('plane_mode must be fixed/ransac3d')
    if not 0 <= params['gray_saturation_max'] <= 255 or not (
            0 <= params['gray_value_min'] < params['gray_value_max'] <= 255):
        raise ValueError('Invalid gray HSV range')
    for key in ('lower_green', 'upper_green'):
        values = params[key]
        if len(values) != 3 or not (0 <= values[0] <= 179 and all(0 <= v <= 255 for v in values[1:])):
            raise ValueError('Invalid OpenCV green HSV range')
    if np.any(np.asarray(params['lower_green']) > params['upper_green']):
        raise ValueError('Reversed green HSV bounds')
    for key in ('sync_max_delta_s', 'image_max_age_s', 'plane_ransac_distance',
                'plane_max_flat_residual_m', 'plane_max_height_spread_m', 'plane_min_xy_span_m',
                'plane_height_hint_tolerance_m', 'depth_region_mad_scale', 'depth_region_min_band_m'):
        if not np.isfinite(params[key]) or params[key] <= 0:
            raise ValueError('Invalid positive parameter: ' + key)
    if not 0 < params['base_plane_min_ratio'] <= 1 or not 0 < params['depth_min_valid_ratio'] <= 1:
        raise ValueError('Inlier/depth ratios must lie in (0,1]')
    if not 0 < params['depth_frame_min_valid_ratio'] <= 1:
        raise ValueError('depth_frame_min_valid_ratio must lie in (0,1]')
    if (params['plane_calibration_frames'] < 3 or params['plane_calibration_max_points'] < 100
            or params['depth_min_valid_pixels'] < 1 or params['depth_erode_px'] < 0
            or params['sync_buffer_size'] < 2 or params['plane_refresh_every'] < 1
            or not 0 <= params['plane_max_tilt_rad'] < np.pi/3
            or not params['plane_max_tilt_rad'] <= params['plane_max_3d_tilt_rad'] < np.pi/3
            or not 0 <= params['plane_empty_gray_max_ratio'] < 1
            or not 0 <= params['rgb_height_min_m'] < params['rgb_height_max_m']
            or not 0 < params['block_height_min'] < params['block_height_max']):
        raise ValueError('Invalid calibration, synchronization or height settings')


def color_masks(color_bgr, params):
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.asarray(params['lower_green'], np.uint8),
                        np.asarray(params['upper_green'], np.uint8)).astype(bool)
    gray = ((hsv[..., 1] <= params['gray_saturation_max'])
            & (hsv[..., 2] >= params['gray_value_min'])
            & (hsv[..., 2] <= params['gray_value_max']) & ~green)
    return green, gray


def calibration_points(color, depth, camera, transform, params):
    roi = params['plane_roi_px']
    if not any(roi):
        raise ValueError('Set plane_roi_px to an explicit empty-belt-only ROI')
    selected = roi_mask(depth.shape, roi)
    check_depth_coverage(depth, params, roi)
    green, gray = color_masks(color, params)
    if np.count_nonzero(gray & selected) / np.count_nonzero(selected) > params['plane_empty_gray_max_ratio']:
        raise ValueError('Gray object/arm detected in calibration ROI; remove objects before calibration')
    points = camera.cloud(depth, transform)
    valid = selected & green & np.isfinite(points).all(axis=2) & np.isfinite(depth) & (depth > 0)
    if params['plane_height_hint_enabled']:
        valid &= np.abs(points[..., 2] - params['plane_height_hint']) <= params['plane_height_hint_tolerance_m']
    return points[valid]


def _workspace_contains(point, params):
    return all(params[f'workspace_{axis}'][0] <= point[i] <= params[f'workspace_{axis}'][1]
               for i, axis in enumerate('xyz'))


def check_depth_coverage(depth, params, roi=None):
    selected = roi_mask(depth.shape, params['roi_px'] if roi is None else roi)
    valid = np.isfinite(depth) & (depth > 0)
    if np.count_nonzero(valid & selected) < params['depth_frame_min_valid_ratio'] * selected.sum():
        raise ValueError('Insufficient valid depth coverage; do not publish this frame as an empty belt')
    return valid


def detect_blocks(color, depth, camera, transform, plane, params):
    """New modes publish vertical base heights, with belt_z + height == centroid Z."""
    if plane is None:
        raise ValueError('An empty-belt plane calibration is required')
    transform = check_transform(transform)
    camera.check_grid(depth, camera.frame_id)
    camera.check_grid(color, camera.frame_id)
    valid = check_depth_coverage(depth, params)
    selected = roi_mask(depth.shape, params['roi_px'])
    mode = params['detection_mode']
    height_mask = None
    if mode in ('depth_plane', 'rgb_depth'):
        heights = plane.height_image(depth, camera, transform)
        height_mask = valid & (heights >= params['block_height_min']) & (heights <= params['block_height_max'])
    if mode == 'depth_plane':
        mask = height_mask & selected
    elif mode in ('rgb', 'rgb_depth'):
        _, gray = color_masks(color, params)
        mask = gray & selected
    else:
        raise ValueError('Use detect_legacy for the legacy_depth baseline')
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    detections = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < params['min_area_px']:
            continue
        # Work within the component bounding box, not a full image per block.
        x, y, w, h = stats[label, :4]
        region = (labels[y:y+h, x:x+w] == label).astype(np.uint8)
        border = params['depth_erode_px']
        if border:
            region = cv2.erode(region, np.ones((2*border+1, 2*border+1), np.uint8),
                               borderType=cv2.BORDER_CONSTANT, borderValue=0)
        region = region.astype(bool)
        usable = region & valid[y:y+h, x:x+w]
        if mode in ('rgb', 'rgb_depth') and (usable.sum() < params['depth_min_valid_pixels'] or
                usable.sum() < params['depth_min_valid_ratio'] * region.sum()):
            raise ValueError('RGB object candidate has insufficient interior depth; frame is ambiguous')
        if mode == 'rgb_depth':
            # Keep the RGB object boundary stable; require sufficient interior depth above the belt.
            usable &= height_mask[y:y+h, x:x+w]
        if (usable.sum() < params['depth_min_valid_pixels'] or
                usable.sum() < params['depth_min_valid_ratio'] * region.sum()):
            continue
        depths = depth[y:y+h, x:x+w][usable]
        median = np.median(depths)
        band = max(params['depth_region_min_band_m'],
                   params['depth_region_mad_scale'] * 1.4826 * np.median(np.abs(depths - median)))
        good = np.abs(depths - median) <= band
        if good.sum() < max(params['depth_min_valid_pixels'], params['depth_min_valid_ratio'] * region.sum()):
            continue
        u, v = centers[label]
        optical = camera.rays([u], [v])[0] * np.median(depths[good])
        point = transform[:3, :3] @ optical + transform[:3, 3]
        belt_z = float(plane.z_at(point[0], point[1]))
        height = float(point[2] - belt_z)
        low, high = ((params['rgb_height_min_m'], params['rgb_height_max_m']) if mode == 'rgb' else
                     (params['block_height_min'], params['block_height_max']))
        if not low <= height <= high or not _workspace_contains(point, params):
            continue
        if not (0 <= u < camera.width and 0 <= v < camera.height):
            continue
        detections.append(dict(u=float(u), v=float(v), centroid_base=point, height=height,
                               width_px=math.sqrt(area), area_px=float(area), belt_z=belt_z))
    return detections, mask


def detect_legacy(color, depth, camera, transform, plane, params, refit=True):
    """Original image-depth RANSAC/segmentation and output semantics for comparison.

    Strict alignment/encoding/time validation is performed by the caller. The
    baseline's optical-axis height is intentionally retained and clearly reported.
    """
    valid = check_depth_coverage(depth, params)
    roi = params['roi_px'] if any(params['roi_px']) else None
    if refit or plane.coeffs is None:
        plane.fit(depth, roi=roi)
    if plane.coeffs is None:
        return [], np.zeros(depth.shape, np.uint8)
    v_grid, u_grid = np.indices(depth.shape)
    plane_z = plane.plane_z(u_grid, v_grid)
    height = plane_z - depth
    fused_components = params.get('legacy_rgb_components', False)
    selected = roi_mask(depth.shape, roi)
    if fused_components:
        # Group an object's visible layers before applying its median-height
        # limit. The old per-pixel upper threshold can split a single stepped
        # object, causing alternating centroids and duplicate track IDs.
        _, gray = color_masks(color, params)
        mask = (gray & valid & selected & (height >= params['block_height_min'])
                & (height <= 2*params['block_height_max'])).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        mask &= selected.astype(np.uint8)
        edges = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), (3, 3), 0), 40, 100)
    else:
        mask = ((height >= params['block_height_min']) & (height <= params['block_height_max'])
                & selected).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    detections = []
    for label in range(1, count):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < params['min_area_px']:
            continue
        u, v = centers[label]
        region = labels == label
        if fused_components:
            # Smooth highlights may have a small stereo depth bias but lack
            # the edges of a solid object. Do not assign them fresh IDs.
            x, y, w, h = stats[label, :4]
            x0, y0 = max(0, x-2), max(0, y-2)
            x1, y1 = min(color.shape[1], x+w+2), min(color.shape[0], y+h+2)
            component = region[y0:y1, x0:x1].astype(np.uint8)
            edge_pixels = np.count_nonzero(edges[y0:y1, x0:x1] &
                cv2.dilate(component, np.ones((5, 5), np.uint8)))
            if edge_pixels < params['legacy_min_edge_support']*2*(w+h):
                continue
        usable = region & valid
        if fused_components and (usable.sum() < params['depth_min_valid_pixels']
                or usable.sum() < params['depth_min_valid_ratio']*area):
            continue
        object_height = float(np.median(height[usable]))
        if fused_components and not params['block_height_min'] <= object_height <= params['block_height_max']:
            continue
        z_top = float(np.median(depth[usable]))
        if not np.isfinite(z_top) or z_top <= 0:
            continue
        ray = camera.rays([u], [v])[0]
        point = transform[:3, :3] @ (ray * z_top) + transform[:3, 3]
        if not _workspace_contains(point, params):
            continue
        z_plane = plane_z[int(round(v)), int(round(u))]
        belt = transform[:3, :3] @ (ray * z_plane) + transform[:3, 3]
        detections.append(dict(u=float(u), v=float(v), centroid_base=point,
                               height=object_height, width_px=math.sqrt(area),
                               area_px=float(area), belt_z=float(belt[2])))
    return detections, mask
