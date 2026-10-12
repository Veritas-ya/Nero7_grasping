"""Offline evaluation CLI; outputs measured metrics with explicit provenance."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import yaml

from ..detection import load_detection_params
from ..base_plane import save_plane, archive_plane
from ..validation import load_dataset, synthetic_dataset, evaluate_dataset


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input', nargs='+', help='capture_rgbd NPZ files; sorted by capture time')
    source.add_argument('--synthetic', action='store_true', help='Mathematical fixture, never real-camera evidence')
    parser.add_argument('--config', required=True, help='perception.yaml')
    parser.add_argument('--annotations', help='JSON {frames: [[{id, centroid_base, height?, velocity_base?}], ...]}')
    parser.add_argument('--output-dir', default='/tmp/nero7_perception_validation')
    parser.add_argument('--plane-mode', choices=['fixed', 'ransac3d'], default='fixed')
    parser.add_argument('--plane-roi', type=int, nargs=4, help='Empty-belt calibration ROI, u0 v0 u1 v1')
    parser.add_argument('--roi', type=int, nargs=4, help='Detection ROI, u0 v0 u1 v1')
    parser.add_argument('--synthetic-tilt-rad', type=float, default=0.)
    parser.add_argument('--synthetic-stress', action='store_true')
    options = parser.parse_args(args)
    cv2.setNumThreads(1)
    params = load_detection_params(options.config)
    output = Path(options.output_dir); output.mkdir(parents=True, exist_ok=True)
    previous_plane = archive_plane(output/'belt_plane.yaml')
    if options.synthetic:
        data, annotations = synthetic_dataset(tilt=options.synthetic_tilt_rad, stress=options.synthetic_stress)
        params.update(workspace_x=[-1., 1.], roi_px=[0, 0, 0, 0],
                      plane_roi_px=[20, 20, 300, 220])
        np.savez_compressed(output/'synthetic.npz', **data)
        (output/'synthetic_annotations.json').write_text(json.dumps(dict(frames=annotations), indent=2))
    else:
        data, annotations = load_dataset(options.input), None
    if options.plane_roi is not None:
        params['plane_roi_px'] = options.plane_roi
    if options.roi is not None:
        params['roi_px'] = options.roi
    if options.annotations:
        annotations = json.loads(Path(options.annotations).read_text())['frames']
    (output/'effective_params.yaml').write_text(yaml.safe_dump(params, sort_keys=False))
    report, planes, camera = evaluate_dataset(data, params, annotations)
    report.update(selected_plane_mode=options.plane_mode,
                  previous_plane_file=previous_plane,
                  plane_file_generated=options.plane_mode in planes)
    (output/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    if options.plane_mode in planes:
        first = int(np.flatnonzero(data['empty'])[0])
        # Synthetic calibration remains inside the synthetic output directory.
        save_plane(output/'belt_plane.yaml', planes[options.plane_mode], camera,
                   data['transforms'][first], str(data['base_frame']), params['plane_roi_px'])
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
