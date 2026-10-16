"""Run the installed driver with local CAN transmit observation enabled.

This diagnostic entry changes only local_loopback and command logging. It does
not rescale feedback, zero the gripper, or enable the arm at startup.
"""
import agx_arm_ctrl.agx_arm_ctrl_single_node as driver
import signal
import threading
import rclpy
from rclpy.signals import SignalHandlerOptions


original_config = driver.create_agx_arm_config
original_move = driver.AgxGripperWrapper.move


def trace_config(**kwargs):
    kwargs['local_loopback'] = True
    kwargs['receive_own_messages'] = False
    return original_config(**kwargs)


def trace_move(self, width=0., force=1.):
    print('[gripper_trace] requested width_m=%s force_N=%s initialized=%s'
          % (width, force, self._initialized), flush=True)
    result = original_move(self, width=width, force=force)
    print('[gripper_trace] SDK move result=%s' % result, flush=True)
    return result


class DiagnosticNode(driver.AgxArmRosNode):
    def _enable_arm(self, enable=True, timeout=5.):
        # The installed driver otherwise re-enables all axes on reconnection,
        # even with auto_enable=false. This diagnostic never resumes them.
        if enable and threading.current_thread() is not threading.main_thread() and not self.auto_enable:
            self.enable_flag = bool(self.agx_arm.get_joint_enable_status(255))
            self.get_logger().warning('诊断模式：通信恢复后不会自动重新使能机械臂。')
            return True
        return super()._enable_arm(enable, timeout)


def main():
    node = None
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    try:
        node = DiagnosticNode()
        # SDK initialization may install process handlers; keep a single owner
        # for ROS shutdown and disconnect the SDK before process exit.
        signal.signal(signal.SIGINT, interrupt)
        signal.signal(signal.SIGTERM, interrupt)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.try_shutdown()
        if node is not None:
            node.agx_arm.disconnect()
            node.destroy_node()


if __name__ == '__main__':
    driver.create_agx_arm_config = trace_config
    driver.AgxGripperWrapper.move = trace_move
    main()
