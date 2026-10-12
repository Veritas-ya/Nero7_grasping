"""Convenience wrapper: record a rosbag of the perception-relevant topics.

Usage:
    ros2 run nero7_perception record_session --ros-args \
        -p output:=/tmp/session_bag -p topics:="[/camera/color/image_raw, /camera/depth/image_raw, /camera/aligned_depth_to_color/image_raw, /perception/tracks, /perception/debug_image]"
"""
import json

import rclpy
from rclpy.node import Node


class RecordSessionNode(Node):
    def __init__(self):
        super().__init__('record_session')
        self._output = self.declare_parameter('output', '/tmp/nero7_session').value
        topics_json = self.declare_parameter(
            'topics',
            '["/camera/color/image_raw", "/camera/aligned_depth_to_color/image_raw", '
            '"/camera/depth/image_raw", "/perception/tracks", "/perception/debug_image"]').value
        self._topics = json.loads(topics_json) if isinstance(topics_json, str) else topics_json
        self.get_logger().info(
            'run this in another terminal:')
        cmd = f'ros2 bag record -o {self._output} ' + ' '.join(self._topics)
        self.get_logger().info(cmd)
        print()
        print('Record command:')
        print(' ', cmd)
        print()
        raise SystemExit(0)


def main(args=None):
    rclpy.init(args=args)
    node = RecordSessionNode()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
