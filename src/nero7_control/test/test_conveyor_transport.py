"""Verify the perception -> generated ROS message -> controller contract."""
from unittest.mock import Mock

import numpy as np
from builtin_interfaces.msg import Time
from rclpy.serialization import serialize_message, deserialize_message
from nero7_msgs.msg import BlockTrackArray

from nero7_control.grasp_controller import TrackView
from nero7_control.conveyor_prediction import motion_at_contact
from nero7_perception.block_detector import BlockDetectorNode
from nero7_perception.block_tracker import BlockTracker
from nero7_perception.conveyor_model import BeltModel


def publisher():
    node = BlockDetectorNode.__new__(BlockDetectorNode)
    node._p = dict(base_frame='base_link')
    node._belt = BeltModel([0., -1.])
    node._pub_tracks = Mock()
    return node


def timestamp(seconds):
    sec, nsec = divmod(int(round(seconds*1e9)), 1_000_000_000)
    return Time(sec=sec, nanosec=nsec)


def trajectory():
    tracker = BlockTracker(BeltModel([0., -1.]), {})
    rng = np.random.default_rng(827)
    start = 1_791_601_000.
    for i in range(91):
        t = start+i/30
        truth = np.array([-.4, .4-.028*i/30, -.065])
        detection = dict(u=500., v=400., centroid_base=truth+np.r_[rng.normal(0, .001, 2), 0.],
                         height=.035, width_px=35., area_px=900., belt_z=-.1)
        tracks = tracker.update([detection], t)
    return tracker, tracks[0], t, truth


def roundtrip(node, tracks, t):
    node._publish_tracks(tracks, timestamp(t))
    sent = node._pub_tracks.publish.call_args.args[0]
    return deserialize_message(serialize_message(sent), BlockTrackArray)


def test_filtered_velocity_and_covariance_survive_serialization():
    _, track, t, truth = trajectory()
    msg = roundtrip(publisher(), [track], t)
    assert msg.header.frame_id == 'base_link'
    assert msg.header.stamp == timestamp(t)
    view = TrackView(msg.tracks[0])
    assert view.speed_valid
    assert abs(view.speed-.028) < .003
    np.testing.assert_allclose(view.motion_covariance, track.motion_covariance)
    np.testing.assert_allclose(view.motion_process_noise, track.motion_process_noise)
    np.testing.assert_allclose(view.observed_position, track.observed_centroid)
    predicted = motion_at_contact(view, t+.02, 3., np.zeros(2), np.zeros((2, 2)))
    expected = truth[:2]+np.array([0., -.028])*3.02
    assert np.linalg.norm(predicted['position']-expected) < .007
    assert np.linalg.eigvalsh(predicted['covariance']).min() > 0


def test_lost_message_keeps_observation_time_and_advances_state_time():
    tracker, track, t, _ = trajectory()
    observed_stamp = track.stamp
    covariance = track.motion_covariance.copy()
    tracker.update([], t+1/30)
    msg = roundtrip(publisher(), tracker.tracks, t+1/30)
    view = TrackView(msg.tracks[0])
    assert view.status == 2 and view.speed_valid
    assert abs(view.stamp-observed_stamp) < 1e-6
    assert view.motion_stamp > view.stamp
    assert np.trace(view.motion_covariance) > np.trace(covariance)
    result = motion_at_contact(view, view.motion_stamp+.01, 1., np.zeros(2),
                               np.zeros((2, 2)), allow_lost=True)
    assert .04 < result['age'] < .05


def test_empty_array_still_carries_detector_image_time():
    msg = roundtrip(publisher(), [], 123.125)
    assert not msg.tracks
    assert msg.header.stamp == Time(sec=123, nanosec=125_000_000)
