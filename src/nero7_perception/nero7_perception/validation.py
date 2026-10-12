"""Reproducible offline calibration/detection/tracking evaluation, never arm control."""
import time
import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

from .geometry import CameraGeometry, check_transform
from .base_plane import EmptyBeltCalibration, plane_context, check_plane_context
from .detection import calibration_points, detect_blocks, detect_legacy
from .conveyor_model import BeltPlane, BeltModel
from .block_tracker import BlockTracker


def load_dataset(paths):
    datasets = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            datasets.append({key: archive[key].copy() for key in archive.files})
    first = datasets[0]
    shared = ('K', 'D', 'distortion_model', 'camera_frame', 'base_frame', 'color_frame', 'depth_frame')
    frame_keys = ('color_bgr', 'depth_m', 'transforms', 'stamps', 'color_stamps', 'empty')
    for data in datasets:
        if any(key not in data for key in shared + frame_keys):
            raise ValueError('Use capture_rgbd NPZ schema; old snapshots lack required metadata')
        if any(not np.array_equal(data[key], first[key]) for key in shared):
            raise ValueError('Capture files use different camera calibrations/frames')
    result = {key: first[key] for key in shared}
    result.update({key: np.concatenate([data[key] for data in datasets], axis=0) for key in frame_keys})
    order = np.argsort(result['stamps'], kind='stable')
    for key in frame_keys:
        result[key] = result[key][order]
    result['provenance'] = [str(data.get('provenance', 'unspecified')) for data in datasets]
    return result


def dataset_camera(data):
    n, height, width = data['depth_m'].shape
    if data['color_bgr'].shape != (n, height, width, 3) or data['color_bgr'].dtype != np.uint8:
        raise ValueError('RGB must be uint8 BGR and share the registered depth grid')
    camera = CameraGeometry(data['K'], width, height, str(data['camera_frame']),
                            tuple(data['D']), str(data['distortion_model']))
    camera.check_grid(data['color_bgr'][0], str(data['color_frame']))
    camera.check_grid(data['depth_m'][0], str(data['depth_frame']))
    if data['transforms'].shape != (n, 4, 4) or any(data[key].shape != (n,) for key in ('stamps', 'color_stamps', 'empty')):
        raise ValueError('Capture metadata lengths do not match images')
    if not np.isfinite(data['stamps']).all() or np.any(np.diff(data['stamps']) <= 0):
        raise ValueError('Depth capture timestamps must be finite, distinct and increasing')
    for transform in data['transforms']:
        check_transform(transform)
    return camera


def calibrate_dataset(data, camera, params, mode):
    indices = np.flatnonzero(data['empty'])
    if len(indices) < params['plane_calibration_frames']:
        raise ValueError('Not enough explicitly labeled empty-belt frames')
    collector = EmptyBeltCalibration(mode, params['plane_calibration_frames'],
        params['plane_ransac_distance'], params['base_plane_min_ratio'], params['plane_max_tilt_rad'],
        params['plane_max_flat_residual_m'], params['plane_max_height_spread_m'], params['plane_min_xy_span_m'],
        params['plane_calibration_max_points'], params['plane_max_3d_tilt_rad'])
    context = plane_context(camera, data['transforms'][indices[0]], str(data['base_frame']))
    for index in indices[:collector.frames]:
        check_plane_context(context, camera, data['transforms'][index], str(data['base_frame']))
        points = calibration_points(data['color_bgr'][index], data['depth_m'][index],
                                    camera, data['transforms'][index], params)
        collector.add(points, data['stamps'][index])
    return collector.fit(), context


def _matches(predictions, truth, max_distance=.04):
    if not predictions or not truth:
        return []
    positions = np.array([p['centroid_base'] for p in predictions])
    target = np.array([g['centroid_base'] for g in truth])
    distances = np.linalg.norm(positions[:, None] - target[None], axis=2)
    rows, columns = linear_sum_assignment(distances)
    return [(int(i), int(j), float(distances[i, j])) for i, j in zip(rows, columns) if distances[i, j] <= max_distance]


def evaluate_dataset(data, params, annotations=None):
    camera = dataset_camera(data)
    data = dict(data)
    data['depth_m'] = data['depth_m'].astype(np.float32, copy=True)
    data['depth_m'][~np.isfinite(data['depth_m']) | (data['depth_m'] <= 0)] = np.nan
    delta = np.abs(data['stamps'] - data['color_stamps'])
    if not np.isfinite(delta).all() or np.any(delta > params['sync_max_delta_s']):
        raise ValueError('RGB-D pair timestamps violate sync_max_delta_s')
    if annotations is not None and len(annotations) != len(data['stamps']):
        raise ValueError('Annotations must have one list per sorted capture frame')
    belt_data = {}
    if params.get('belt_file'):
        with open(params['belt_file']) as stream:
            belt_data = yaml.safe_load(stream) or {}
    belt = BeltModel.from_yaml(belt_data)
    report = dict(provenance=data.get('provenance', 'unspecified'), frames=len(data['stamps']),
                  empty_frames=int(data['empty'].sum()), image_size=[camera.width, camera.height],
                  geometry_units='m', annotated=annotations is not None,
                  timing_scope='offline detection only, excluding ROS/registration/tracking; not live FPS',
                  plane_calibration={}, modes={})
    planes, contexts = {}, {}
    for mode in ('fixed', 'ransac3d'):
        started = time.perf_counter()
        try:
            plane, context = calibrate_dataset(data, camera, params, mode)
            planes[mode], contexts[mode] = plane, context
            report['plane_calibration'][mode] = dict(plane.diagnostics, normal=plane.normal.tolist(),
                offset=plane.offset, calibration_ms=(time.perf_counter()-started)*1000)
        except ValueError as error:
            report['plane_calibration'][mode] = dict(error=str(error))
    combinations = [('legacy_depth', None)] + [(mode, plane) for plane in ('fixed', 'ransac3d')
        for mode in ('depth_plane', 'rgb', 'rgb_depth')]
    for detection_mode, plane_mode in combinations:
        name = detection_mode if plane_mode is None else f'{detection_mode}/{plane_mode}'
        if plane_mode is not None and plane_mode not in planes:
            report['modes'][name] = dict(skipped='Plane calibration rejected')
            continue
        settings = dict(params, detection_mode=detection_mode)
        legacy = BeltPlane(params['plane_ransac_distance'], params['plane_ransac_min_ratio'])
        speed_cfg = dict(window_s=params.get('speed_window_s', .6), min_samples=params.get('speed_min_samples', 8),
                         min_span=params.get('speed_min_span', .3))
        tracker = BlockTracker(belt, speed_cfg, max_vel=params.get('tracker_max_vel', .15),
            gate_radius=params.get('tracker_gate_radius', .03), confirm_frames=params.get('tracker_confirm_frames', 2),
            lost_frames=params.get('tracker_lost_frames', 6), lost_timeout=params.get('tracker_lost_timeout', .5),
            motion_cfg={key.removeprefix('kalman_'): value for key, value in params.items() if key.startswith('kalman_')})
        times, positions, heights, speeds, counts = [], [], [], [], []
        true_positive = false_positive = false_negative = empty_false_positive = switches = qualified = 0
        rejected_frames, rejection_reasons = 0, {}
        ids, last_ids = set(), {}
        for index, stamp in enumerate(data['stamps']):
            started = time.perf_counter()
            rejected = False
            try:
                if plane_mode is None:
                    detections, _ = detect_legacy(data['color_bgr'][index], data['depth_m'][index], camera,
                        data['transforms'][index], legacy, settings, refit=index % params['plane_refresh_every'] == 0)
                else:
                    check_plane_context(contexts[plane_mode], camera, data['transforms'][index], str(data['base_frame']))
                    detections, _ = detect_blocks(data['color_bgr'][index], data['depth_m'][index], camera,
                        data['transforms'][index], planes[plane_mode], settings)
            except ValueError as error:
                detections, rejected = [], True
                rejected_frames += 1
                rejection_reasons[str(error)] = rejection_reasons.get(str(error), 0)+1
            times.append((time.perf_counter()-started)*1000)
            counts.append(len(detections))
            if data['empty'][index]:
                empty_false_positive += len(detections)
            tracks = tracker.update(detections, float(stamp)) if not rejected else []
            observed = [tr for tr in tracks if tr.status != 2]
            ids.update(tr.id for tr in observed)
            qualified += sum(tr.speed_valid for tr in observed)
            if annotations is None:
                continue
            truth = annotations[index]
            matches = _matches(detections, truth)
            true_positive += len(matches)
            false_positive += len(detections) - len(matches)
            false_negative += len(truth) - len(matches)
            for i, j, error in matches:
                positions.append(error)
                if 'height' in truth[j]:
                    heights.append(abs(detections[i]['height'] - truth[j]['height']))
            tracked = [dict(centroid_base=tr.centroid) for tr in observed]
            for i, j, _ in _matches(tracked, truth):
                tr, target = observed[i], truth[j]
                key = str(target['id'])
                if key in last_ids and last_ids[key] != tr.id:
                    switches += 1
                last_ids[key] = tr.id
                if tr.speed_valid and 'velocity_base' in target:
                    speeds.append(abs(tr.speed - np.asarray(target['velocity_base'])[:2] @ belt.d_hat))
        report['modes'][name] = dict(detection_ms_mean=float(np.mean(times)), detection_ms_p95=float(np.percentile(times, 95)),
            rejected_frames=rejected_frames, rejection_reasons=rejection_reasons,
            offline_detection_fps=1000/float(np.mean(times)), detection_count_min=min(counts), detection_count_max=max(counts),
            empty_false_positives=empty_false_positive, observed_unique_track_ids=len(ids), qualified_speed_samples=qualified,
            true_positives=true_positive if annotations is not None else None,
            false_positives=false_positive if annotations is not None else None,
            false_negatives=false_negative if annotations is not None else None,
            precision=(true_positive/(true_positive+false_positive) if true_positive+false_positive else None) if annotations is not None else None,
            recall=(true_positive/(true_positive+false_negative) if true_positive+false_negative else None) if annotations is not None else None,
            position_rmse_m=float(np.sqrt(np.mean(np.square(positions)))) if positions else None,
            height_mae_m=float(np.mean(heights)) if heights else None,
            speed_mae_m_s=float(np.mean(speeds)) if speeds else None,
            id_switches=switches if annotations is not None else None)
    return report, planes, camera


def synthetic_dataset(frames=60, empty_frames=15, width=320, height=240, tilt=.0, seed=24, stress=False):
    """Ray-rendered horizontal/sloped belt, moving cuboid, controlled axial noise.

    This is a mathematical fixture, not Orbbec or Gazebo measurements.
    """
    rng = np.random.default_rng(seed)
    camera = CameraGeometry(np.array([[300., 0, width/2], [0, 300., height/2], [0, 0, 1]]),
                            width, height, 'camera_color_optical_frame')
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler('y', .45).as_matrix() @ np.diag([1., -1., -1.])
    transform[:3, 3] = [-.4, .2, .65]
    v, u = np.indices((height, width))
    rays = camera.rays(u, v).reshape(height, width, 3) @ transform[:3, :3].T
    belt_h = -.10286622820279956
    normal = np.array([-np.tan(tilt), 0., 1.])
    denom = np.einsum('...i,i->...', rays, normal)
    depths, colors, annotations = [], [], []
    for frame in range(frames):
        depth = (belt_h - normal @ transform[:3, 3]) / denom
        color = np.full((height, width, 3), [50, 140, 50], np.uint8)
        objects = []
        if frame >= empty_frames:
            # Place block near the optical center, moving with the real belt direction.
            center_xy = np.array([-.74, .2 - .028*(frame-empty_frames)/30.])
            top_h = belt_h + np.tan(tilt)*center_xy[0] + .04
            object_depth = (top_h-transform[2, 3]) / rays[..., 2]
            surface = transform[:3, 3] + rays*object_depth[..., None]
            body = (np.abs(surface[..., 0]-center_xy[0]) <= .035) & (np.abs(surface[..., 1]-center_xy[1]) <= .025)
            depth[body] = object_depth[body]
            color[body] = [145, 145, 145]
            objects = [dict(id=1, centroid_base=[*center_xy, top_h], height=.04, velocity_base=[0., -.028, 0.])]
            if stress:
                # A gray paint mark on the belt plus a controlled 6 mm local axial bias.
                # Ground truth contains only the actual cuboid, not the paint mark.
                color[25:60, 230:275] = 145
                depth[25:60, 230:275] -= .006
                outliers = body & (rng.random(depth.shape) < .1)
                depth[outliers] += .035
        depth += rng.normal(0, .0008, depth.shape)
        invalid = rng.random(depth.shape) < .01
        depth[invalid] = np.nan
        depths.append(depth.astype(np.float32)); colors.append(color); annotations.append(objects)
    data = dict(color_bgr=np.stack(colors), depth_m=np.stack(depths), K=camera.K, D=np.array([]),
        distortion_model=np.array('plumb_bob'), camera_frame=np.array(camera.frame_id),
        color_frame=np.array(camera.frame_id), depth_frame=np.array(camera.frame_id), base_frame=np.array('base_link'),
        transforms=np.repeat(transform[None], frames, axis=0), stamps=100.+np.arange(frames)/30,
        color_stamps=100.+np.arange(frames)/30, empty=np.arange(frames)<empty_frames,
        provenance='synthetic ray rendering; axial sigma=0.8 mm; 1% depth dropout'
                   + ('; stress: gray belt mark, -6 mm local axial bias, 10% object depth outliers' if stress else ''))
    return data, annotations
