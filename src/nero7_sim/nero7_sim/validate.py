"""Independent Gazebo evidence collector; never sends motion commands."""
import argparse
import json
import time
from pathlib import Path
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String
from gazebo_msgs.msg import ModelStates
from nero7_msgs.msg import BlockTrackArray


class Validation(Node):
    def __init__(self, output):
        super().__init__('simulation_validation')
        self.output = Path(output); self.output.mkdir(parents=True, exist_ok=True)
        self.bridge = CvBridge(); self.image = None
        self.record = dict(states=[], grasp_events=[], vision=[], object_z=[], passed=False)
        self.create_subscription(String, '/sim/grasp/state', self.state, 10)
        self.create_subscription(String, '/sim/scene/grasp_event', self.event, 10)
        self.create_subscription(BlockTrackArray, '/sim/perception/tracks', self.tracks, 10)
        self.create_subscription(ModelStates, '/sim/scene/ground_truth', self.truth, 10)
        self.create_subscription(Image, '/sim/camera/color/image_raw', self.color, qos_profile_sensor_data)

    def color(self, msg):
        self.image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def state(self, msg):
        if not self.record['states'] or self.record['states'][-1] != msg.data:
            self.record['states'].append(msg.data); print(msg.data, flush=True)
        if msg.data.endswith('_HOLD') and self.image is not None:
            cv2.imwrite(str(self.output/'lift.png'), self.image)

    def event(self, msg):
        self.record['grasp_events'].append(msg.data)

    def tracks(self, msg):
        for track in msg.tracks:
            if track.status == 1 and track.speed_valid:
                self.record['vision'].append(dict(stamp=track.stamp.sec+track.stamp.nanosec*1e-9,
                    speed=track.speed, x=track.centroid_base.x, y=track.centroid_base.y))

    def truth(self, msg):
        for name, pose in zip(msg.name, msg.pose):
            if name == 'example_block': self.record['object_z'].append(pose.position.z)

    def result(self):
        z = self.record['object_z']
        lifted = bool(z and max(z)-z[0] > .07)
        completed = any(s in ('CONVEYOR_DONE', 'STATIC_DONE') for s in self.record['states'])
        attached = 'ATTACHED_GEOMETRIC_CONTACT' in self.record['grasp_events']
        self.record.update(passed=bool(lifted and completed and attached), lifted=lifted,
                           object_lift_m=max(z)-z[0] if z else 0.)
        if self.record['vision']:
            self.record['median_visual_speed_m_s'] = float(np.median([v['speed'] for v in self.record['vision']]))
        (self.output/'validation.json').write_text(json.dumps(self.record, indent=2))
        if self.image is not None: cv2.imwrite(str(self.output/'final.png'), self.image)
        return self.record['passed']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timeout', type=float, default=180.)
    parser.add_argument('--output', default='/tmp/nero7_gazebo/validation')
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args); node = Validation(args.output)
    started = time.monotonic()
    try:
        while time.monotonic()-started < args.timeout:
            rclpy.spin_once(node, timeout_sec=.1)
            if any(s in ('CONVEYOR_DONE', 'STATIC_DONE', 'CONVEYOR_ABORT', 'STATIC_ABORT') for s in node.record['states']):
                break
        passed = node.result(); print('SIMULATION_PASS' if passed else 'SIMULATION_INCOMPLETE', flush=True)
    finally:
        node.destroy_node(); rclpy.try_shutdown()
    raise SystemExit(0 if passed else 1)
