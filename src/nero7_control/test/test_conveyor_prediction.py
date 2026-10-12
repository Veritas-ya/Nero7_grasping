import unittest
import numpy as np
from nero7_control.conveyor_prediction import fit_velocity, contact_prediction, contact_delay


class PredictionTest(unittest.TestCase):
    def test_velocity_uses_image_timestamps_and_metric_positions(self):
        samples = [(1791545758.+i/30, [-.4, .55-.028*i/30]) for i in range(45)]
        velocity, residual = fit_velocity(samples, [0, -1])
        np.testing.assert_allclose(velocity, [0, -.028], atol=1e-7)
        self.assertLess(residual, 1e-7)

    def test_gate_contact_accounts_for_image_age_and_grasp_point_offset(self):
        predicted = contact_prediction([-.4, .295], 100., 100.1,
                                       [0, -.028], 3., [.005, .0018])
        np.testing.assert_allclose(predicted, [-.395, .21], atol=1e-10)

    def test_stale_image_blocks_prediction(self):
        with self.assertRaises(ValueError):
            contact_prediction([0, 0], 100., 100.3, [0, -.028], 2., [0, 0])

    def test_stationary_reverse_noisy_or_duplicate_observations_rejected(self):
        for kind in ['stationary', 'reverse', 'noisy', 'duplicate']:
            samples = []
            for i in range(45):
                t = i/30 if kind != 'duplicate' else 1.
                y = .028*i/30 if kind == 'reverse' else (0 if kind == 'stationary' else -.028*i/30)
                x = .02*(-1)**i if kind == 'noisy' else 0
                samples.append((t, [x, y]))
            with self.assertRaises(ValueError):
                fit_velocity(samples, [0, -1])

    def test_contact_precedes_empty_close_completion(self):
        delay = contact_delay(1., .070, .0205, .024)
        self.assertGreater(delay, .8)
        self.assertLess(delay, 1.)


if __name__ == '__main__':
    unittest.main()
