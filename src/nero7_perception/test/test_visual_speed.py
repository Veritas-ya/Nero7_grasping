import unittest
import numpy as np

from nero7_perception.conveyor_model import BeltModel, SpeedEstimator
from nero7_perception.block_tracker import BlockTracker, STATUS_LOST


class VisualSpeedTest(unittest.TestCase):
    def fitted(self, velocity=.05):
        fit = SpeedEstimator()
        for i in range(20):
            fit.update(1791543778. + i/30, -.36 + velocity*i/30)
        return fit

    def test_metric_speed_with_large_image_timestamps(self):
        fit = self.fitted()
        self.assertTrue(fit.ok)
        self.assertAlmostEqual(fit.v, .05, places=6)

    def test_reverse_motion_preserves_sign(self):
        fit = self.fitted(-.04)
        self.assertTrue(fit.ok)
        self.assertAlmostEqual(fit.v, -.04, places=6)

    def test_stationary_target_has_zero_speed(self):
        fit = self.fitted(0.)
        self.assertTrue(fit.ok)
        self.assertAlmostEqual(fit.v, 0., places=8)

    def test_repeated_frame_cannot_converge(self):
        fit = SpeedEstimator()
        for _ in range(30):
            fit.update(10., .1)
        self.assertEqual(len(fit.window), 1)
        self.assertFalse(fit.ok)

    def test_gap_invalidates_old_speed(self):
        fit = self.fitted()
        fit.update(fit.window[-1][0]+1., fit.window[-1][1])
        self.assertFalse(fit.ok)
        self.assertIsNone(fit.v)

    def test_clock_reset_and_position_jump_require_new_fit(self):
        for clock_reset in [True, False]:
            fit = self.fitted()
            t, s = fit.window[-1]
            fit.update(t-2 if clock_reset else t+.03, s if clock_reset else s+.2)
            self.assertFalse(fit.ok)
            self.assertEqual(len(fit.window), 1)

    def test_noisy_observations_are_not_valid_speed(self):
        fit = SpeedEstimator()
        for i in range(20):
            fit.update(i/30, .05*i/30 + .02*(-1)**i)
        self.assertFalse(fit.ok)

    def test_invalid_position_clears_fit(self):
        fit = self.fitted()
        fit.update(fit.window[-1][0]+.03, np.nan)
        self.assertFalse(fit.ok)
        self.assertIsNone(fit.v)

    def test_tracker_projects_motion_and_invalidates_lost_track(self):
        direction = np.array([.6, .8])
        tracker = BlockTracker(BeltModel(direction), {}, lost_frames=6)
        for i in range(20):
            xy = np.array([-.36, .2]) + direction * .05*i/30
            det = dict(u=400., v=400., centroid_base=np.r_[xy, -.06],
                       height=.04, width_px=30., area_px=800.)
            tracks = tracker.update([det], i/30)
        self.assertEqual(len(tracks), 1)
        self.assertTrue(tracks[0].speed_ok)
        self.assertAlmostEqual(tracks[0].speed, .05, delta=1e-4)
        last_stamp = tracks[0].stamp
        accepted_count = tracks[0].motion.accepted_count
        covariance = tracks[0].motion_covariance.copy()
        tracks = tracker.update([], 20/30)
        self.assertEqual(tracks[0].status, STATUS_LOST)
        self.assertEqual(tracks[0].stamp, last_stamp)
        self.assertTrue(tracks[0].speed_valid)
        self.assertEqual(tracks[0].motion.accepted_count, accepted_count)
        self.assertGreater(tracks[0].motion_stamp, last_stamp)
        self.assertGreater(np.trace(tracks[0].motion_covariance), np.trace(covariance))
        for i in range(21, 25):
            tracks = tracker.update([], i/30)
        self.assertFalse(tracks[0].speed_valid)
        self.assertIsNone(tracks[0].speed)
        for i in range(25, 27):
            tracks = tracker.update([], i/30)
        self.assertEqual(tracks, [])


if __name__ == '__main__':
    unittest.main()
