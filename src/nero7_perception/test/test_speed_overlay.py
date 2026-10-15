"""Display actual object speed without turning missing estimates into zeros."""
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
from builtin_interfaces.msg import Time
from cv_bridge import CvBridge

from nero7_perception.block_detector import BlockDetectorNode
from nero7_perception.block_tracker import BlockTracker
from nero7_perception.conveyor_model import BeltModel
from nero7_perception.speed_overlay import draw_speed_overlay, speed_label


def track(**changes):
    values = dict(id=7, status=1, speed_valid=True, velocity=[-.03, .04, 0.],
                  speed=-.03, u_px=500., v_px=300.)
    return SimpleNamespace(**dict(values, **changes))


def test_magnitude_includes_cross_belt_motion_and_is_nonnegative():
    assert speed_label(track()) == 'ID 7  speed=0.050 m/s'
    assert speed_label(track(velocity=[-.028, 0., 0.])) == 'ID 7  speed=0.028 m/s'


def test_qualified_stationary_speed_is_zero_but_unqualified_speed_is_unavailable():
    assert speed_label(track(velocity=[0., 0., 0.])) == 'ID 7  speed=0.000 m/s'
    assert speed_label(track(speed_valid=False)) == 'ID 7  speed=-- (WAITING)'
    assert speed_label(track(status=2)) == 'ID 7  speed=-- (LOST)'
    assert speed_label(track(velocity=[np.nan, 0., 0.])) == 'ID 7  speed=-- (INVALID)'


def test_distinct_blocks_have_their_own_speed_and_edge_labels_remain_visible():
    image = np.zeros((480, 640, 3), np.uint8)
    tracks = [track(u_px=635., v_px=2.),
              track(id=8, velocity=[0., .028, 0.], u_px=12., v_px=470.)]
    with patch('nero7_perception.speed_overlay.cv2.putText', wraps=cv2.putText) as put:
        result = draw_speed_overlay(image, tracks)
    assert [call.args[1] for call in put.call_args_list] == [
        'ID 7  speed=0.050 m/s', 'ID 8  speed=0.028 m/s']
    for call in put.call_args_list:
        _, text, (x, y), font, scale, _, thickness = call.args[:7]
        (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
        assert x >= 0 and x + width < 640
        assert y - height >= 0 and y + baseline < 480
    assert result.any()
    assert not image.any()


def test_empty_belt_has_no_numeric_speed():
    with patch('nero7_perception.speed_overlay.cv2.putText') as put:
        draw_speed_overlay(np.zeros((480, 640, 3), np.uint8), [])
    assert put.call_args.args[1] == 'No tracked blocks - speed unavailable'


def test_detection_error_hides_even_previously_valid_object_speed():
    with patch('nero7_perception.speed_overlay.cv2.putText') as put:
        draw_speed_overlay(np.zeros((480, 640, 3), np.uint8), [track()],
                           error='Legacy plane fit unavailable')
    labels = [call.args[1] for call in put.call_args_list]
    assert labels == ['Detection unavailable - speed unavailable', 'Legacy plane fit unavailable']
    assert not any('0.050' in label or 'No tracked blocks' in label for label in labels)


def test_real_tracker_labels_update_and_lost_track_hides_old_speed():
    tracker = BlockTracker(BeltModel([0., -1.]), {})
    for i in range(45):
        detections = [dict(u=250., v=250., centroid_base=[-.4, .4-.028*i/30, -.065],
                           height=.035, width_px=35., area_px=900.)]
        tracks = tracker.update(detections, i/30)
    assert speed_label(tracks[0]) == 'ID 0  speed=0.028 m/s'
    tracker.update([], 45/30)
    assert speed_label(tracks[0]) == 'ID 0  speed=-- (LOST)'


def test_debug_image_keeps_capture_timestamp_and_optical_frame():
    detector = BlockDetectorNode.__new__(BlockDetectorNode)
    detector._p = dict(camera_optical_frame='camera_color_optical_frame', debug_max_width=640)
    detector._color = np.zeros((480, 640, 3), np.uint8)
    detector._bridge = CvBridge()
    detector._pub_debug = Mock()
    stamp = Time(sec=123, nanosec=456)
    detector._publish_debug([track()], stamp)
    message = detector._pub_debug.publish.call_args.args[0]
    assert message.header.stamp == stamp
    assert message.header.frame_id == 'camera_color_optical_frame'
    assert message.encoding == 'bgr8'
    assert detector._bridge.imgmsg_to_cv2(message, 'bgr8').any()


def test_full_resolution_preview_preserves_camera_pixels_and_marker_coordinates():
    detector = BlockDetectorNode.__new__(BlockDetectorNode)
    detector._p = dict(camera_optical_frame='camera_color_optical_frame', debug_max_width=0)
    detector._color = np.full((720, 1280, 3), 80, np.uint8)
    detector._bridge, detector._pub_debug = CvBridge(), Mock()
    detector._publish_debug([track(u_px=1000., v_px=500.)], Time(sec=123))
    msg = detector._pub_debug.publish.call_args.args[0]
    assert (msg.width, msg.height) == (1280, 720)
    image = detector._bridge.imgmsg_to_cv2(msg, 'bgr8')
    np.testing.assert_array_equal(image[600:], detector._color[600:])
    assert np.any(image[490:510, 990:1010] != 80)


def test_close_objects_keep_nonoverlapping_labels_with_their_own_ids():
    rectangles = []
    with patch('nero7_perception.speed_overlay.cv2.putText', wraps=cv2.putText) as put:
        draw_speed_overlay(np.zeros((720, 1280, 3), np.uint8),
                           [track(u_px=480., v_px=400.), track(id=8, u_px=650., v_px=410.)])
    for call in put.call_args_list:
        _, text, (x, y), font, scale, _, thickness = call.args[:7]
        (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
        rectangles.append((x-3, y-h-3, x+w+3, y+baseline+3))
    first, second = rectangles
    assert first[2] < second[0] or second[2] < first[0] or first[3] < second[1] or second[3] < first[1]
