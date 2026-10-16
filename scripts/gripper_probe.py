"""Read feedback, or send one bounded gripper-only command through ROS.

No arm targets, resets, zeroing, SDK connection, or persistent settings are sent.
"""
import argparse
import json
import math
from pathlib import Path
import socket
import struct
import time

import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from agx_arm_msgs.msg import AgxArmStatus, GripperStatus
from std_srvs.srv import SetBool, Empty


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--target-mm', type=float, default=50.)
    args = parser.parse_args()
    if not math.isfinite(args.target_mm) or not 50 <= args.target_mm <= 70:
        parser.error('诊断目标限制为 50～70 mm')
    rclpy.init()
    node = rclpy.create_node('local_gripper_probe')
    received = {}
    subscriptions = []
    bus = None
    gate_owned = False
    report = dict(arm_target_sent=False, gripper_command_sent=False)
    for key, kind, topic in (
            ('joints', JointState, '/feedback/joint_states'),
            ('status', AgxArmStatus, '/feedback/arm_status'),
            ('gripper', GripperStatus, '/feedback/gripper_status')):
        subscriptions.append(node.create_subscription(kind, topic,
            lambda msg, key=key: received.update({key: (time.monotonic(), msg)}),
            qos_profile_sensor_data))

    def fresh():
        if len(received) != 3 or any(time.monotonic()-at > .3 for at, _ in received.values()):
            return False
        now = node.get_clock().now().nanoseconds*1e-9
        return all(0 <= now-msg.header.stamp.sec-msg.header.stamp.nanosec*1e-9 <= .3
                   for _, msg in (received['joints'], received['gripper']))

    def call(client, request):
        if not client.wait_for_service(timeout_sec=1.):
            raise RuntimeError('Service unavailable: '+client.srv_name)
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=1.)
        if not future.done():
            raise RuntimeError('Service timed out: '+client.srv_name)
        response = future.result()
        if hasattr(response, 'success') and not response.success:
            raise RuntimeError(str(response))

    try:
        deadline = time.monotonic()+5
        while time.monotonic() < deadline and not fresh():
            rclpy.spin_once(node, timeout_sec=.02)
        if not fresh():
            raise RuntimeError('Fresh joint/status/gripper feedback unavailable')
        grip = received['gripper'][1]
        status = received['status'][1]
        report.update(initial_reported_width_m=grip.width,
                      gripper_enabled=grip.driver_enable_status,
                      homing_status=grip.homing_status,
                      arm_motion_status=status.motion_status,
                      target_mm=args.target_mm)
        if not args.execute:
            return
        if 'grasp_controller' in [name for name, _ in node.get_node_names_and_namespaces()]:
            raise RuntimeError('请先停止抓取节点，保留机械臂驱动')
        if status.motion_status not in (0, 1) or status.err_status or status.arm_status in (1, 5, 7, 9, 10):
            raise RuntimeError('Stopped, fault-free arm required')
        if any(status.joint_angle_limit) or any(status.communication_status_joint):
            raise RuntimeError('Arm limit/communication fault')
        if not grip.driver_enable_status or any(getattr(grip, key) for key in (
                'voltage_too_low', 'motor_overheating', 'driver_overcurrent',
                'driver_overheating', 'sensor_status', 'driver_error_status')):
            raise RuntimeError('Enabled, fault-free gripper required')
        js = received['joints'][1]
        names = ['joint%d' % i for i in range(1, 8)]
        if any(js.name.count(name) != 1 for name in names):
            raise RuntimeError('Seven named arm joints required')
        seed = [js.position[js.name.index(name)] for name in names]
        if not all(math.isfinite(q) for q in seed):
            raise RuntimeError('Invalid arm joint seed')
        # Status 1 means target not reached; confirm actual seven-axis stillness
        # independently before a gripper-only test. No arm path is permitted.
        deadline = time.monotonic()+1.
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.01)
            if not fresh():
                raise RuntimeError('Feedback stale while confirming stationary arm')
            js = received['joints'][1]
            actual = [js.position[js.name.index(name)] for name in names]
            if any(not math.isfinite(a) or abs(a-b) > .002 for a, b in zip(actual, seed)):
                raise RuntimeError('Arm is not stationary; no gripper command sent')
        publisher = node.create_publisher(JointState, '/control/joint_states', 1)
        gate = node.create_client(SetBool, '/control_enable')
        stop = node.create_client(Empty, '/emergency_stop')
        deadline = time.monotonic()+2
        while publisher.get_subscription_count() != 1 and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.02)
        if publisher.get_subscription_count() != 1 or node.count_publishers('/control/joint_states') != 1:
            raise RuntimeError('Exclusive command publisher and exactly one driver required')
        if any(node.count_publishers('/control/'+topic) for topic in (
                'move_j', 'move_p', 'move_l', 'move_js', 'move_c', 'move_mit', 'move_cpv')):
            raise RuntimeError('Another arm command publisher is present')
        bus = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        filters = b''.join(struct.pack('=II', cid, 0x7ff) for cid in (0x159, 0x2a8))
        bus.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER, filters)
        bus.bind(('can0',))
        bus.setblocking(False)
        gate_owned = True
        call(gate, SetBool.Request(data=True))
        command = JointState()
        command.header.stamp = node.get_clock().now().to_msg()
        command.name = ['gripper']
        command.position = [args.target_mm/1000.]
        command.effort = [1.]
        publisher.publish(command)
        report['gripper_command_sent'] = True
        report.update(can_command_frames=[], feedback_raw_min=None, feedback_raw_max=None,
                      reported_width_min_m=grip.width, reported_width_max_m=grip.width)
        deadline = time.monotonic()+4
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.01)
            if not fresh():
                raise RuntimeError('Feedback lost during gripper probe')
            js = received['joints'][1]
            actual = [js.position[js.name.index(name)] for name in names]
            if any(not math.isfinite(a) or abs(a-b) > .03 for a, b in zip(actual, seed)):
                call(stop, Empty.Request())
                raise RuntimeError('Unexpected arm movement; stop requested')
            width = received['gripper'][1].width
            if not math.isfinite(width):
                raise RuntimeError('Invalid gripper width feedback')
            report['reported_width_min_m'] = min(report['reported_width_min_m'], width)
            report['reported_width_max_m'] = max(report['reported_width_max_m'], width)
            while True:
                try:
                    raw = bus.recv(16)
                except BlockingIOError:
                    break
                cid, length, data = struct.unpack('=IB3x8s', raw)
                if cid & 0x7ff == 0x159:
                    report['can_command_frames'].append(data[:length].hex())
                else:
                    value = int.from_bytes(data[:4], 'big', signed=True)
                    for key, function in (('feedback_raw_min', min), ('feedback_raw_max', max)):
                        report[key] = value if report[key] is None else function(report[key], value)
        report['final_reported_width_m'] = received['gripper'][1].width
        report['expected_can_command'] = (int(round(args.target_mm*1000)).to_bytes(4, 'big')
                                           + bytes.fromhex('03e80100')).hex()
        report['expected_command_observed'] = report['expected_can_command'] in report['can_command_frames']
        report['can_observation_note'] = '本机发送抓包需驱动启用 local_loopback；未观察到帧本身不能证明未发送。'
    except Exception as error:
        report['error'] = str(error)
        report['feedback_ages'] = {}
        now = node.get_clock().now().nanoseconds*1e-9
        for key, (at, message) in received.items():
            ages = dict(receipt_age_s=time.monotonic()-at)
            if hasattr(message, 'header'):
                stamp = message.header.stamp
                ages['source_age_s'] = now-stamp.sec-stamp.nanosec*1e-9
            report['feedback_ages'][key] = ages
    finally:
        if gate_owned:
            try:
                call(gate, SetBool.Request(data=False))
                report['control_gate_closed'] = True
            except Exception as error:
                report['control_gate_close_error'] = str(error)
        if bus is not None:
            bus.close()
        path = Path(__file__).resolve().parents[1]/'validation/latest-branch'/('gripper-ros-probe-'+time.strftime('%Y%m%d-%H%M%S')+'.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        report['report_file'] = str(path)
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
        node.destroy_node()
        rclpy.shutdown()
    return 1 if 'error' in report else 0


if __name__ == '__main__':
    raise SystemExit(main())
