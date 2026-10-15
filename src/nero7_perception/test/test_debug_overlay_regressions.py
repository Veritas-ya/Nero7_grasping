from types import SimpleNamespace as NS
import numpy as np
from nero7_perception.speed_overlay import draw_speed_overlay, speed_label


def test_downsized_debug_scales_markers_without_changing_tracking_coordinates():
    image = np.zeros((360, 640, 3), np.uint8)
    track = NS(id=7, u_px=800., v_px=400., status=1, speed_valid=True,
               velocity=np.array([0., -.028, 0.]))
    original = image.copy()
    debug = draw_speed_overlay(image, [track], pixel_scale=.5)
    assert np.any(debug[190:210, 390:410])
    np.testing.assert_array_equal(image, original)
    assert track.u_px == 800. and track.v_px == 400.
    assert '0.028 m/s' in speed_label(track)


def test_waiting_and_lost_do_not_display_a_numeric_speed():
    track = NS(id=7, u_px=200., v_px=100., status=1, speed_valid=False,
               velocity=np.array([0., -.028, 0.]))
    assert 'WAITING' in speed_label(track) and '0.028' not in speed_label(track)
    track.status = 2
    assert 'LOST' in speed_label(track) and '0.028' not in speed_label(track)
