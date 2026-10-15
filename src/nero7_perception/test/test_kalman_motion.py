"""Motion, timestamp, uncertainty and identity regressions without ROS/hardware."""
import unittest

import numpy as np

from nero7_perception.block_tracker import (
    BlockTracker, STATUS_CONFIRMED, STATUS_LOST,
)
from nero7_perception.conveyor_model import BeltModel
from nero7_perception.kalman_motion import KalmanMotion, transition_and_noise


def detection(xy, u=400., height=.04):
    return dict(u=u, v=400., centroid_base=np.r_[xy, -.06],
                height=height, belt_z=-.10, width_px=30., area_px=800.)


class KalmanMotionTest(unittest.TestCase):
    def fitted(self, velocity=(.02, -.028), duration=2.0, stamp=1791543778.):
        position = np.array([-.36, .2])
        velocity = np.asarray(velocity)
        motion = KalmanMotion(position, stamp)
        for i in range(1, int(duration * 30) + 1):
            self.assertTrue(motion.update(position + velocity * i / 30., stamp + i / 30.))
        return motion

    def test_metric_velocity_and_stationary_from_independent_observations(self):
        for velocity in ((.02, -.028), (-.04, 0.), (0., 0.)):
            motion = self.fitted(velocity)
            self.assertTrue(motion.speed_valid)
            np.testing.assert_allclose(motion.state[2:], velocity, atol=5e-5)

    def test_kalman_reduces_position_noise_and_reports_velocity_uncertainty(self):
        rng = np.random.default_rng(721)
        velocity = np.array([0., -.028])
        position = np.array([-.36, .2])
        motion = KalmanMotion(position, 10., belt_direction=velocity)
        filtered_errors, observed_errors = [], []
        for i in range(1, 91):
            truth = position + velocity * i / 30.
            observed = truth + rng.normal(0., .002, 2)
            accepted = motion.update(observed, 10. + i / 30.)
            if accepted and i >= 30:
                filtered_errors.append(motion.state[:2] - truth)
                observed_errors.append(observed - truth)
        self.assertTrue(motion.speed_valid)
        self.assertLess(np.sqrt(np.mean(np.square(filtered_errors))),
                        np.sqrt(np.mean(np.square(observed_errors))) * .7)
        np.testing.assert_allclose(motion.state[2:], velocity, atol=.003)
        self.assertTrue(np.all(np.linalg.eigvalsh(motion.covariance) > 0))

    def test_repeated_or_out_of_order_frame_cannot_converge(self):
        motion = KalmanMotion([0., 0.], 10.)
        for i in range(50):
            self.assertFalse(motion.update([i * .01, 0.], 10.))
            self.assertFalse(motion.update([i * .01, 0.], 9.))
        self.assertEqual(motion.accepted_count, 1)
        self.assertEqual(motion.observation_stamp, 10.)
        self.assertFalse(motion.speed_valid)
        np.testing.assert_array_equal(motion.state, np.zeros(4))

    def test_outlier_does_not_change_raw_observation_or_velocity(self):
        motion = self.fitted()
        raw, old_stamp = motion.observed_xy.copy(), motion.observation_stamp
        velocity = motion.state[2:].copy()
        self.assertFalse(motion.update(raw + [.020, -.001], old_stamp + 1 / 30.))
        np.testing.assert_array_equal(motion.observed_xy, raw)
        np.testing.assert_array_equal(motion.state[2:], velocity)
        self.assertEqual(motion.observation_stamp, old_stamp)
        self.assertEqual(motion.rejected_count, 1)

    def test_nan_measurement_does_not_refresh_stamp(self):
        motion = self.fitted()
        stamp = motion.observation_stamp
        self.assertFalse(motion.update([np.nan, 0.], stamp + .03))
        self.assertEqual(motion.observation_stamp, stamp)

    def test_coasting_grows_covariance_and_stale_observation_invalidates_speed(self):
        motion = self.fitted()
        covariance = motion.covariance.copy()
        stamp = motion.observation_stamp
        motion.predict(stamp + .10)
        self.assertTrue(motion.speed_valid)
        self.assertEqual(motion.observation_stamp, stamp)
        self.assertGreater(np.trace(motion.covariance[:2, :2]),
                           np.trace(covariance[:2, :2]))
        motion.predict(stamp + .16)
        self.assertFalse(motion.speed_valid)

    def test_long_gap_requires_new_qualification(self):
        motion = self.fitted()
        stamp = motion.observation_stamp + .5
        predicted, _ = motion.forecast(stamp)
        self.assertTrue(motion.update(predicted[:2], stamp))
        self.assertEqual(len(motion.accepted_stamps), 1)
        self.assertFalse(motion.speed_valid)

    def test_forecast_has_process_noise_and_does_not_mutate_filter(self):
        motion = self.fitted()
        state, covariance = motion.state.copy(), motion.covariance.copy()
        stamp = motion.state_stamp
        dt = 3.2
        transition, process = transition_and_noise(dt, motion.process_noise)
        forecast_state, forecast_covariance = motion.forecast(stamp + dt)
        np.testing.assert_allclose(forecast_state, transition @ state)
        np.testing.assert_allclose(forecast_covariance,
                                   transition @ covariance @ transition.T + process)
        self.assertGreater(process[0, 0], 0.)
        np.testing.assert_array_equal(motion.state, state)
        np.testing.assert_array_equal(motion.covariance, covariance)
        self.assertEqual(motion.state_stamp, stamp)

    def test_belt_direction_rotates_anisotropic_acceleration_density(self):
        direction = np.array([.6, .8])
        cross = np.array([-.8, .6])
        motion = KalmanMotion([0., 0.], 10., belt_direction=direction)
        density = motion.process_noise
        self.assertAlmostEqual(direction @ density @ direction, .002 ** 2)
        self.assertAlmostEqual(cross @ density @ cross, .0005 ** 2)
        _, noise = transition_and_noise(3.2, density)
        self.assertAlmostEqual(direction @ noise[:2, :2] @ direction,
                               .002 ** 2 * 3.2 ** 3 / 3.)
        self.assertAlmostEqual(cross @ noise[:2, :2] @ cross,
                               .0005 ** 2 * 3.2 ** 3 / 3.)

    def test_covariance_threshold_can_withhold_valid_speed(self):
        motion = KalmanMotion([0., 0.], 0., max_velocity_std=.00001)
        for i in range(1, 31):
            motion.update([0., -.028 * i / 30.], i / 30.)
        self.assertFalse(motion.speed_valid)


class KalmanTrackerTest(unittest.TestCase):
    def fitted_tracker(self, direction=(0., -1.), speed=.028, **kwargs):
        direction = np.asarray(direction)
        tracker = BlockTracker(BeltModel(direction), {}, **kwargs)
        start = np.array([-.36, .20])
        for i in range(61):
            tracker.update([detection(start + direction * speed * i / 30.)],
                           10. + i / 30.)
        self.assertEqual(len(tracker.tracks), 1)
        return tracker

    def test_signed_projection_and_raw_depth_metadata(self):
        tracker = self.fitted_tracker(direction=(.6, .8), speed=-.028)
        track = tracker.tracks[0]
        self.assertEqual(track.status, STATUS_CONFIRMED)
        self.assertTrue(track.speed_valid)
        self.assertAlmostEqual(track.speed, -.028, places=4)
        self.assertEqual(track.velocity[2], 0.)
        self.assertEqual(track.centroid[2], track.observed_centroid[2])
        self.assertEqual(track.motion_stamp, track.stamp)
        self.assertEqual(track.motion_covariance.shape, (4, 4))
        self.assertEqual(track.motion_process_noise.shape, (2, 2))

    def test_missing_frames_predict_and_recover_same_id(self):
        tracker = self.fitted_tracker()
        original = tracker.tracks[0]
        old_stamp = original.stamp
        old_raw = original.observed_centroid.copy()
        old_xy = original.centroid[:2].copy()
        count = original.motion.accepted_count
        for i in (1, 2, 3):
            tracker.update([], old_stamp + i / 30.)
        track = tracker.tracks[0]
        self.assertEqual(track.status, STATUS_LOST)
        self.assertTrue(track.speed_valid)
        self.assertEqual(track.stamp, old_stamp)
        self.assertEqual(track.motion.accepted_count, count)
        np.testing.assert_array_equal(track.observed_centroid, old_raw)
        np.testing.assert_allclose(track.centroid[:2], old_xy + track.velocity[:2] * .10)
        t = old_stamp + 4 / 30.
        tracker.update([detection(old_raw[:2] + [0., -.028 * 4 / 30.])], t)
        self.assertEqual(len(tracker.tracks), 1)
        self.assertEqual(tracker.tracks[0].id, original.id)
        self.assertEqual(tracker.tracks[0].status, STATUS_CONFIRMED)
        self.assertEqual(tracker.tracks[0].stamp, t)

    def test_stale_lost_speed_is_unavailable_then_track_expires(self):
        tracker = self.fitted_tracker(lost_frames=10)
        stamp = tracker.tracks[0].stamp
        tracker.update([], stamp + .16)
        self.assertFalse(tracker.tracks[0].speed_valid)
        self.assertIsNone(tracker.tracks[0].speed)
        self.assertEqual(tracker.update([], stamp + .51), [])

    def test_confirmed_id_uses_elapsed_time_and_gap_expiry(self):
        tracker = self.fitted_tracker(lost_frames=3)
        stamp = tracker.tracks[0].stamp
        for i in range(1, 4):
            self.assertEqual(len(tracker.update([], stamp + i / 30.)), 1)
        self.assertEqual(len(tracker.update([], stamp + 4 / 30.)), 1)
        self.assertEqual(tracker.update([], stamp + .51), [])
        tracker = self.fitted_tracker()
        old_id = tracker.tracks[0].id
        raw = tracker.tracks[0].observed_centroid[:2].copy()
        stamp = tracker.tracks[0].stamp
        tracks = tracker.update([detection(raw + [0., -.028])], stamp + 1.)
        self.assertEqual(len(tracks), 1)
        self.assertNotEqual(tracks[0].id, old_id)
        self.assertFalse(tracks[0].speed_valid)

    def test_duplicate_and_out_of_order_frames_do_not_age_or_create_tracks(self):
        tracker = self.fitted_tracker()
        track = tracker.tracks[0]
        raw = track.observed_centroid.copy()
        count = track.motion.accepted_count
        for stamp in (track.stamp, track.stamp - 1.):
            tracker.update([], stamp)
            tracker.update([detection([0., 0.])], stamp)
        self.assertEqual(len(tracker.tracks), 1)
        self.assertEqual(track.misses, 0)
        self.assertEqual(track.motion.accepted_count, count)
        np.testing.assert_array_equal(track.observed_centroid, raw)

    def test_innovation_outlier_is_not_accepted_or_spawned_as_duplicate(self):
        tracker = self.fitted_tracker()
        track = tracker.tracks[0]
        raw, stamp = track.observed_centroid.copy(), track.stamp
        tracker.update([detection(raw[:2] + [.020, -.001], height=.06)], stamp + 1 / 30.)
        self.assertEqual(len(tracker.tracks), 1)
        self.assertEqual(track.status, STATUS_LOST)
        self.assertEqual(track.stamp, stamp)
        self.assertEqual(track.height, .04)
        np.testing.assert_array_equal(track.observed_centroid, raw)

    def test_association_is_one_to_one_and_independent_of_detection_order(self):
        tracker = BlockTracker(BeltModel((0., -1.)), {})
        first, second = np.array([-.36, .20]), np.array([-.42, .20])
        tracks = tracker.update([detection(first, u=400.), detection(second, u=450.)], 10.)
        ids = [track.id for track in tracks]
        tracks = tracker.update([detection(second + [0., -.001], u=451.),
                                 detection(first + [0., -.001], u=401.)], 10. + 1 / 30.)
        self.assertEqual(len(tracks), 2)
        self.assertEqual([track.id for track in tracks], ids)
        self.assertEqual([track.u_px for track in tracks], [401., 451.])
        self.assertEqual([track.hits for track in tracks], [2, 2])


if __name__ == '__main__':
    unittest.main()
