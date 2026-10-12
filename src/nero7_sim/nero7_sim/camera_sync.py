"""Publish paired rendered frames after /sim/clock has reached their time.

Gazebo camera callbacks and the clock publisher run independently. Preserve
the measured timestamps/data; a short bounded delay avoids future-stamped
frames reaching the hardware controller's strict freshness guards.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo


class CameraSync(Node):
    def __init__(self):
        super().__init__('camera_sync')
        self.frames = {}
        self.last_stamp = -1
        self.image_publishers = {kind: self.create_publisher(Image, f'/sim/camera/{kind}/image_raw', 5)
                           for kind in ('color', 'depth')}
        self.info = self.create_publisher(CameraInfo, '/sim/camera/color/camera_info', 5)
        for kind in ('color', 'depth'):
            self.create_subscription(Image, f'/sim/camera/raw/{kind}/image_raw',
                                     lambda msg, kind=kind: self.receive(kind, msg), 5)
        self.create_subscription(CameraInfo, '/sim/camera/raw/color/camera_info',
                                 self.info.publish, 5)
        self.create_timer(.01, self.flush)

    def receive(self, kind, msg):
        stamp = msg.header.stamp.sec*1_000_000_000+msg.header.stamp.nanosec
        if stamp > self.last_stamp:
            self.frames.setdefault(stamp, {})[kind] = msg

    def flush(self):
        now = self.get_clock().now().nanoseconds
        ready = [stamp for stamp, frame in self.frames.items()
                 if len(frame) == 2 and 30_000_000 <= now-stamp <= 150_000_000]
        if ready:
            stamp = max(ready)
            for kind, msg in self.frames[stamp].items():
                self.image_publishers[kind].publish(msg)
            self.last_stamp = stamp
        self.frames = {stamp: frame for stamp, frame in self.frames.items()
                       if stamp > self.last_stamp and now-stamp <= 150_000_000}


def main(args=None):
    rclpy.init(args=args); node = CameraSync()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.try_shutdown()
