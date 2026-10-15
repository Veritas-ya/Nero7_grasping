"""Keep layered gray objects intact without joining belt reflections."""
import numpy as np

from nero7_perception.conveyor_model import BeltPlane
from nero7_perception.detection import DETECTION_DEFAULTS, detect_legacy
from nero7_perception.geometry import CameraGeometry


def scene():
    color = np.full((120, 160, 3), [30, 150, 30], np.uint8)
    depth = np.full((120, 160), .9)
    for x in [30, 90]:
        color[40:75, x:x+35] = 150
        depth[40:75, x:x+35] = .85
    depth[54:60, 30:65] = .82  # Higher surface splits the original depth mask.
    color[48:54, 65:90] = 150  # Gray reflection on the belt, connecting RGB only.
    camera = CameraGeometry([[200, 0, 80], [0, 200, 60], [0, 0, 1]], 160, 120, 'optical')
    plane = BeltPlane()
    plane.coeffs = np.array([0., 0., .9])
    params = dict(DETECTION_DEFAULTS, roi_px=[0, 0, 0, 0], block_height_min=.008,
                  block_height_max=.06, min_area_px=200, workspace_x=[-1, 1],
                  workspace_y=[-1, 1], workspace_z=[0, 2])
    return color, depth, camera, plane, params


def test_layered_object_is_one_detection_and_reflection_does_not_merge_neighbors():
    color, depth, camera, plane, params = scene()
    original, _ = detect_legacy(color, depth, camera, np.eye(4), plane, params, refit=False)
    assert len(original) == 3
    fused, _ = detect_legacy(color, depth, camera, np.eye(4), plane,
                              dict(params, legacy_rgb_components=True), refit=False)
    assert len(fused) == 2
    np.testing.assert_allclose(sorted(d['u'] for d in fused), [47., 107.], atol=.2)


def test_fused_component_rejects_tall_object_and_invalid_depth():
    color, depth, camera, plane, params = scene()
    depth[40:75, 30:65] = .8  # Component median exceeds the normal object-height limit.
    depth[40:75, 90:125] = np.nan
    detections, _ = detect_legacy(color, depth, camera, np.eye(4), plane,
                                   dict(params, legacy_rgb_components=True), refit=False)
    assert detections == []


def test_smooth_glare_depth_bias_cannot_create_an_extra_object_id():
    color, depth, camera, plane, params = scene()
    color[:] = 150  # A soft, uniform highlight has no object contour.
    depth[:] = .9
    depth[40:75, 40:90] = .885  # Stereo bias can otherwise look like a low object.
    detections, _ = detect_legacy(color, depth, camera, np.eye(4), plane,
                                   dict(params, legacy_rgb_components=True), refit=False)
    assert detections == []
