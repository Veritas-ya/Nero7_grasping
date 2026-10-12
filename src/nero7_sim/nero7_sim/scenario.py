"""Spawn an object only after vision and (optional) verified timing are ready."""
import json
from pathlib import Path
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger


class Scenario(Node):
    def __init__(self):
        super().__init__('scenario')
        self.timing = self.declare_parameter('timing_file', '').value
        self.delay = float(self.declare_parameter('delay', 2.).value)
        self.image_time = None; self.future = None; self.finished = False
        self.create_subscription(Image, '/sim/camera/color/image_raw', self.image, 5)
        self.client = self.create_client(Trigger, '/sim/belt/spawn')
        self.timer = self.create_timer(.1, self.tick)

    def image(self, msg):
        if self.image_time is None: self.image_time = self.get_clock().now().nanoseconds*1e-9

    def tick(self):
        if self.finished or self.image_time is None: return
        if self.timing:
            try:
                report = json.loads(Path(self.timing).read_text())
            except (OSError, ValueError): return
            if not report.get('timing_verified') or not any(e['state']=='CONVEYOR_CALIB_DONE' for e in report.get('events', [])):
                return
        if self.get_clock().now().nanoseconds*1e-9-self.image_time < self.delay: return
        if self.future is None and self.client.service_is_ready():
            self.future = self.client.call_async(Trigger.Request())
        elif self.future is not None and self.future.done():
            self.finished = True
            self.get_logger().info(str(self.future.result().message))


def main(args=None):
    rclpy.init(args=args); node = Scenario()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.try_shutdown()
