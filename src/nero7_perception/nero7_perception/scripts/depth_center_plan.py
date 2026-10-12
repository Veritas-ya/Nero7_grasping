"""Read-only depth center preview. Never publishes arm commands."""
import json
import time
import numpy as np
import yaml
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from nero7_perception.depth_center import estimate_depth_centers, short_side_orientations


def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('depth_center_plan')
    calib = node.declare_parameter('calibration_file', '/home/ubuntu22/Nero7_grasping/src/nero7_bringup/config/calibration.yaml').value
    roi = node.declare_parameter('roi_px', [180, 355, 1060, 475]).value
    topic = node.declare_parameter('depth_topic', '/camera/depth/image_raw').value
    output = node.declare_parameter('output_file', '/tmp/nero7_depth_center_plan.json').value
    closing_axis = node.declare_parameter('closing_axis_flange', [0.0, 0.0, 1.0]).value
    min_surface_height = node.declare_parameter('min_surface_height', .008).value
    horizontal = node.declare_parameter('horizontal_closing_axis', False).value
    frames = []
    infos = []
    node.create_subscription(Image, topic, lambda m: frames.append(m), qos_profile_sensor_data)
    node.create_subscription(CameraInfo, '/camera/color/camera_info', lambda m: infos.append(m), qos_profile_sensor_data)
    try:
        deadline = time.monotonic()+6
        while (len(frames) < 5 or not infos) and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.1)
        if len(frames) < 5 or not infos:
            raise RuntimeError('Missing depth frames or color intrinsics')
        info = infos[-1]
        K = np.array(info.k).reshape(3, 3)
        with open(calib) as f:
            config = yaml.safe_load(f)
        T = np.array(config['camera_to_base_cv'])
        results = []
        bridge = CvBridge()
        for frame in frames[-5:]:
            if frame.header.frame_id != 'camera_color_optical_frame':
                raise RuntimeError('Depth must be registered to the color optical frame')
            if (frame.width, frame.height) != (info.width, info.height):
                raise RuntimeError('Depth and color intrinsics resolutions differ')
            img = bridge.imgmsg_to_cv2(frame, 'passthrough')
            if frame.encoding == '16UC1':
                img = img.astype(float)/1000
            elif frame.encoding != '32FC1':
                raise RuntimeError('Unsupported depth encoding: '+frame.encoding)
            results.append(estimate_depth_centers(img, K, T, roi,
                min_surface_height=min_surface_height))
        result = results[-1]
        # Require an unambiguous object and stable estimates over all frames.
        stable = all(len(r['candidates']) == 1 for r in results)
        if stable:
            centers = np.array([r['candidates'][0]['center_base'] for r in results])
            stable = bool(np.max(np.linalg.norm(centers-centers.mean(axis=0), axis=1)) < .005)
        result['stable_single_object'] = stable
        for candidate in result['candidates']:
            candidate['orientation_options'] = short_side_orientations(
                config['grasp_rpy'], closing_axis,
                candidate['closing_direction_xy'], horizontal=horizontal)
            candidate['closing_axis_flange'] = list(closing_axis)
        result['generated_at_unix'] = time.time()
        result['preview_only'] = True
        with open(output, 'w') as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result, indent=2), flush=True)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
