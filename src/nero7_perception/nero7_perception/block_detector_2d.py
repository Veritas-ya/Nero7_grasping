#!/usr/bin/env python3

import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class BlockDetector2D(Node):

    def __init__(self):
        super().__init__("block_detector_2d")

        # ============================================================
        # 参数
        # ============================================================

        self.declare_parameter(
            "color_topic",
            "/camera/color/image_raw"
        )

        self.declare_parameter(
            "debug_topic",
            "/perception/debug_image_2d"
        )

        # 最小物块面积
        self.declare_parameter(
            "min_area",
            500
        )

        # 最大物块面积
        self.declare_parameter(
            "max_area",
            100000
        )

        # 绿色 HSV 范围
        self.declare_parameter(
            "green_h_min",
            35
        )

        self.declare_parameter(
            "green_h_max",
            90
        )

        self.declare_parameter(
            "green_s_min",
            40
        )

        self.declare_parameter(
            "green_s_max",
            255
        )

        self.declare_parameter(
            "green_v_min",
            40
        )

        self.declare_parameter(
            "green_v_max",
            255
        )

        # ============================================================
        # 获取参数
        # ============================================================

        self.color_topic = self.get_parameter(
            "color_topic"
        ).value

        self.debug_topic = self.get_parameter(
            "debug_topic"
        ).value

        self.min_area = self.get_parameter(
            "min_area"
        ).value

        self.max_area = self.get_parameter(
            "max_area"
        ).value

        self.green_h_min = self.get_parameter(
            "green_h_min"
        ).value

        self.green_h_max = self.get_parameter(
            "green_h_max"
        ).value

        self.green_s_min = self.get_parameter(
            "green_s_min"
        ).value

        self.green_s_max = self.get_parameter(
            "green_s_max"
        ).value

        self.green_v_min = self.get_parameter(
            "green_v_min"
        ).value

        self.green_v_max = self.get_parameter(
            "green_v_max"
        ).value

        # ============================================================
        # ROS 工具
        # ============================================================

        self.bridge = CvBridge()

        # ============================================================
        # Subscriber
        # ============================================================

        self.image_subscription = self.create_subscription(
            Image,
            self.color_topic,
            self.image_callback,
            10
        )

        # ============================================================
        # Publisher
        # ============================================================

        self.debug_publisher = self.create_publisher(
            Image,
            self.debug_topic,
            10
        )

        self.get_logger().info(
            "BlockDetector2D started."
        )

        self.get_logger().info(
            f"Color topic: {self.color_topic}"
        )

        self.get_logger().info(
            f"Debug topic: {self.debug_topic}"
        )

    # ================================================================
    # 图像回调
    # ================================================================

    def image_callback(self, msg):

        try:

            # ROS Image → OpenCV BGR 图像
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8"
            )

        except Exception as e:

            self.get_logger().error(
                f"Failed to convert image: {e}"
            )

            return

        # 处理图像
        debug_image = self.detect_blocks(frame)

        # OpenCV → ROS Image
        try:

            debug_msg = self.bridge.cv2_to_imgmsg(
                debug_image,
                encoding="bgr8"
            )

            debug_msg.header = msg.header

            self.debug_publisher.publish(
                debug_msg
            )

        except Exception as e:

            self.get_logger().error(
                f"Failed to publish debug image: {e}"
            )

    # ================================================================
    # 物块检测
    # ================================================================

    def detect_blocks(self, frame):

        # ------------------------------------------------------------
        # 原始图像
        # ------------------------------------------------------------

        debug_image = frame.copy()

        # ------------------------------------------------------------
        # 转 HSV
        # ------------------------------------------------------------

        hsv = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2HSV
        )

        # ------------------------------------------------------------
        # 绿色传送带的 HSV 范围
        # ------------------------------------------------------------

        lower_green = np.array([
            self.green_h_min,
            self.green_s_min,
            self.green_v_min
        ])

        upper_green = np.array([
            self.green_h_max,
            self.green_s_max,
            self.green_v_max
        ])

        # ------------------------------------------------------------
        # 找绿色区域
        # ------------------------------------------------------------

        green_mask = cv2.inRange(
            hsv,
            lower_green,
            upper_green
        )

        # ------------------------------------------------------------
        # 非绿色区域作为物块候选
        # ------------------------------------------------------------

        block_mask = cv2.bitwise_not(
            green_mask
        )

        # ============================================================
        # 形态学处理
        # ============================================================

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5)
        )

        # 去除小噪声
        block_mask = cv2.morphologyEx(
            block_mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1
        )

        # 填充物块内部的小孔洞
        block_mask = cv2.morphologyEx(
            block_mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2
        )

        # ============================================================
        # 连通域分析
        # ============================================================

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            block_mask,
            connectivity=8
        )

        block_count = 0

        # ------------------------------------------------------------
        # 遍历所有连通区域
        # ------------------------------------------------------------

        for i in range(1, num_labels):

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]

            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            area = stats[i, cv2.CC_STAT_AREA]

            # --------------------------------------------------------
            # 面积过滤
            # --------------------------------------------------------

            if area < self.min_area:
                continue

            if area > self.max_area:
                continue

            # --------------------------------------------------------
            # 计算中心
            # --------------------------------------------------------

            cx = int(centroids[i][0])
            cy = int(centroids[i][1])

            block_count += 1

            # --------------------------------------------------------
            # 绘制 bounding box
            # --------------------------------------------------------

            cv2.rectangle(
                debug_image,
                (x, y),
                (x + w, y + h),
                (0, 255, 255),
                10
            )

            # --------------------------------------------------------
            # 绘制中心点
            # --------------------------------------------------------

            cv2.circle(
                debug_image,
                (cx, cy),
                6,
                (255, 0, 0),
                -1
            )

            # --------------------------------------------------------
            # 显示编号
            # --------------------------------------------------------

            cv2.putText(
                debug_image,
                f"Block {block_count}",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            # --------------------------------------------------------
            # 显示中心坐标
            # --------------------------------------------------------

            cv2.putText(
                debug_image,
                f"({cx}, {cy})",
                (x, y + h + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

            # --------------------------------------------------------
            # 输出到终端
            # --------------------------------------------------------

            self.get_logger().info(
                f"Block {block_count}: "
                f"center=({cx}, {cy}), "
                f"area={area}"
            )

        # ============================================================
        # 左上角显示检测数量
        # ============================================================

        cv2.putText(
            debug_image,
            f"Blocks: {block_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )

        # ============================================================
        # 可视化绿色区域轮廓
        # ============================================================

        contours, _ = cv2.findContours(
            green_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(
            debug_image,
            contours,
            -1,
            (0, 255, 0),
            1
        )

        return debug_image


# ====================================================================
# main
# ====================================================================

def main(args=None):

    rclpy.init(args=args)

    node = BlockDetector2D()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":
    main()