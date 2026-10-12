"""URDF FK and bounded IK used to preflight static grasp poses."""
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares


class Kinematics:
    def __init__(self, path):
        root = ET.parse(path).getroot()
        self.joints = []
        limits = []
        for i in range(1, 8):
            joint = root.find("joint[@name='joint%d']" % i)
            origin = joint.find('origin')
            xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
            rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')
            T = np.eye(4)
            T[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
            T[:3, 3] = xyz
            axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
            limit = joint.find('limit')
            limits.append([float(limit.get('lower'))+.02, float(limit.get('upper'))-.02])
            self.joints.append((T, axis))
        self.lower, self.upper = np.array(limits).T

    def fk(self, q):
        T = np.eye(4)
        for angle, (origin, axis) in zip(q, self.joints):
            A = np.eye(4)
            A[:3, :3] = Rotation.from_rotvec(axis*angle).as_matrix()
            T = T @ origin @ A
        return T

    def solve(self, position, rotation, seed):
        def residual(q):
            T = self.fk(q)
            return np.r_[T[:3, 3]-position,
                         (rotation.inv()*Rotation.from_matrix(T[:3, :3])).as_rotvec()*.25]
        sol = least_squares(residual, np.clip(seed, self.lower, self.upper),
                            bounds=(self.lower, self.upper), max_nfev=800)
        if np.linalg.norm(residual(sol.x)) > 1e-4:
            raise ValueError('No verified IK for network grasp pose')
        return sol.x


def yaw_orientations(reference_rpy, axis_flange, yaw):
    reference = Rotation.from_euler('xyz', reference_rpy)
    axis = np.asarray(axis_flange, float)
    axis /= np.linalg.norm(axis)
    current = reference.apply(axis)
    if np.linalg.norm(current[:2]) < .5:
        raise ValueError('Gripper closing axis must be approximately horizontal')
    delta = (yaw-np.arctan2(current[1], current[0])+np.pi/2) % np.pi-np.pi/2
    for angle in [delta, delta+np.pi if delta < 0 else delta-np.pi]:
        rot = Rotation.from_euler('z', angle)*reference
        direction = rot.apply(axis)
        level = np.r_[direction[:2]/np.linalg.norm(direction[:2]), 0.]
        correction, _ = Rotation.align_vectors([level], [direction])
        yield correction*rot
