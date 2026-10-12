import unittest
import numpy as np
from nero7_perception.depth_center import estimate_depth_centers, short_side_orientations
from scipy.spatial.transform import Rotation


class DepthCenterTest(unittest.TestCase):
    def setUp(self):
        self.K = np.array([[500., 0, 100], [0, 500., 100], [0, 0, 1]])
        self.T = np.diag([1., -1., -1., 1.])
        self.T[2, 3] = 1
        self.depth = np.ones((200, 200))

    def estimate(self):
        return estimate_depth_centers(self.depth, self.K, self.T, [0, 0, 200, 200])

    def test_flat_belt_has_no_object(self):
        self.assertEqual(self.estimate()['candidates'], [])

    def test_body_center_ignores_small_tall_protrusion(self):
        self.depth[85:115, 80:120] = .96
        self.depth[96:104, 96:104] = .93
        plan = self.estimate()['candidates'][0]
        self.assertAlmostEqual(plan['body_height'], .04, places=5)
        self.assertAlmostEqual(plan['center_base'][2], .02, places=5)
        self.assertLess(np.linalg.norm(plan['center_base'][:2]), .003)
        self.assertTrue(plan['feasible'])
        self.assertLess(plan['opening'], .07)

    def test_oversize_object_is_rejected(self):
        self.depth[60:140, 60:140] = .96
        self.assertFalse(self.estimate()['candidates'][0]['feasible'])

    def test_invalid_depth_is_rejected(self):
        self.depth[:] = np.nan
        with self.assertRaises(ValueError):
            self.estimate()

    def test_orientation_aligns_short_side_and_preserves_approach(self):
        rpy = [1.179487379, 1.235243032, 2.725334799]
        original = Rotation.from_euler('xyz', rpy)
        for option in short_side_orientations(rpy, [0, 0, 1], [1, 0]):
            axis = np.array(option['closing_axis_base'])
            self.assertAlmostEqual(axis[1], 0, places=7)
            self.assertAlmostEqual(axis[2], original.apply([0, 0, 1])[2], places=7)
            result = Rotation.from_euler('xyz', option['rpy'])
            self.assertAlmostEqual(result.apply([1, 0, 0])[2], original.apply([1, 0, 0])[2], places=7)

    def test_degenerate_closing_axis_is_rejected(self):
        with self.assertRaises(ValueError):
            short_side_orientations([0, 0, 0], [0, 0, 1], [1, 0])

    def test_upper_region_target_stays_inside_selected_height_band(self):
        self.depth[85:115, 80:120] = .955
        result = estimate_depth_centers(self.depth, self.K, self.T,
            [0, 0, 200, 200], min_surface_height=.035)
        plan = result['candidates'][0]
        self.assertAlmostEqual(plan['fingertip_target_base'][2], .037, places=5)
        self.assertTrue(plan['feasible'])

    def test_level_closing_axis_has_equal_fingertip_heights(self):
        for option in short_side_orientations([1.179, 1.235, 2.725], [0, 0, 1], [1, 0], horizontal=True):
            self.assertAlmostEqual(option['closing_axis_base'][2], 0, places=7)
            self.assertAlmostEqual(option['closing_axis_base'][1], 0, places=7)


if __name__ == '__main__':
    unittest.main()
