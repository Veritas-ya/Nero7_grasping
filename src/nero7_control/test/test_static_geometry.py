import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from nero7_control.kinematics import Kinematics, yaw_orientations
from nero7_control.grasp_inference import GraspInference


class GeometryTest(unittest.TestCase):
    def test_flange_fk_matches_recorded_hardware_pose(self):
        kin = Kinematics('/home/ubuntu22/nero_ws/install/nero_description/share/nero_description/urdf/nero_description.urdf')
        q = [-.4449193, .276879, -.4254938, 2.0089263, 2.5207441, .5516288, -.2842443]
        actual = np.array([-.197111, .229580, .268072])
        self.assertLess(np.linalg.norm(kin.fk(q)[:3, 3]-actual), .001)
        goal = kin.fk(q)
        solved = kin.solve(goal[:3, 3], Rotation.from_matrix(goal[:3, :3]), q)
        self.assertLess(np.linalg.norm(kin.fk(solved)-goal), 1e-4)

    def test_network_yaw_aligns_mounted_gripper_closing_axis(self):
        for rotation in yaw_orientations([1.179487379, 1.235243032, 2.725334799], [0, 0, 1], .7):
            axis = rotation.apply([0, 0, 1])
            desired = np.array([np.cos(.7), np.sin(.7)])
            self.assertAlmostEqual(abs(axis[:2] @ desired), 1., places=6)
            self.assertAlmostEqual(axis[2], 0., places=6)

    def inference(self, depth):
        inference = GraspInference.__new__(GraspInference)
        inference._device = 'cpu'
        inference._q_threshold = .25
        inference._target_radius_px = 60.
        outputs = {k: torch.zeros((1, 1, 224, 224)) for k in ['pos', 'cos', 'sin', 'width']}
        outputs['pos'] += 1
        outputs['sin'] += 1
        outputs['width'] += .2
        inference._model = SimpleNamespace(predict=lambda x: outputs)
        req = dict(rgb=np.zeros((300, 300, 3), np.uint8), depth=depth,
                   u=150, v=150, K=np.array([[500., 0, 150], [0, 500., 150], [0, 0, 1]]),
                   stamp=1., tf=dict(R=np.eye(3), t=np.array([.1, .2, .3])))
        grasp = SimpleNamespace(center=(112, 112), angle=np.pi/4, length=30.)
        with patch('nero7_control.grasp_inference.detect_grasps', return_value=[grasp]):
            return inference._infer(req)

    def test_anticlockwise_image_angle_and_base_translation(self):
        result = self.inference(np.full((300, 300), 1000, np.uint16))
        np.testing.assert_allclose(result['center'], [.1, .2, 1.3])
        self.assertAlmostEqual(result['yaw'], -np.pi/4, places=6)

    def test_missing_grasp_depth_does_not_use_background_depth(self):
        depth = np.full((300, 300), 1000, np.uint16)
        depth[145:156, 145:156] = 0
        with np.testing.suppress_warnings() as warnings:
            warnings.filter(RuntimeWarning)
            self.assertEqual(self.inference(depth)['status'], 1)


if __name__ == '__main__':
    unittest.main()
