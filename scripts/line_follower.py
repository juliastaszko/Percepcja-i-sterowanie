#!/usr/bin/env python3

import math
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image


class LineFollower(Node):
    def __init__(self):
        super().__init__("roboracer_line_follower")

        # ROS topics
        self.declare_parameter("image_topic", "/camera/color/image_raw")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")

        # Image processing
        self.declare_parameter("roi_top", 0.55)       # fraction of image height
        self.declare_parameter("roi_bottom", 0.96)
        self.declare_parameter("hsv_h_low", 15)
        self.declare_parameter("hsv_h_high", 40)
        self.declare_parameter("hsv_s_low", 90)
        self.declare_parameter("hsv_v_low", 90)
        self.declare_parameter("min_line_area", 35.0)
        self.declare_parameter("morph_kernel", 3)

        # Steering controller. Error is normalized to [-1, 1]:
        # positive means line is left of image centre -> turn left.
        self.declare_parameter("kp", 1.25)
        self.declare_parameter("ki", 0.00)
        self.declare_parameter("kd", 0.08)
        self.declare_parameter("max_yaw_rate", 1.4)
        self.declare_parameter("integral_limit", 1.0)

        # Speed control
        self.declare_parameter("base_speed", 0.30)
        self.declare_parameter("min_speed", 0.12)
        self.declare_parameter("max_speed", 0.40)
        self.declare_parameter("speed_reduction", 0.55)

        # Safety / diagnostics
        self.declare_parameter("image_timeout", 1.0)
        self.declare_parameter("log_interval", 1.0)
        self.declare_parameter("show_debug", False)

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.cmd_topic = str(self.get_parameter("cmd_vel_topic").value)

        self.roi_top = float(self.get_parameter("roi_top").value)
        self.roi_bottom = float(self.get_parameter("roi_bottom").value)
        if not (0.0 <= self.roi_top < self.roi_bottom <= 1.0):
            raise ValueError("Require 0 <= roi_top < roi_bottom <= 1.")

        self.h_low = int(self.get_parameter("hsv_h_low").value)
        self.h_high = int(self.get_parameter("hsv_h_high").value)
        self.s_low = int(self.get_parameter("hsv_s_low").value)
        self.v_low = int(self.get_parameter("hsv_v_low").value)
        self.min_line_area = float(self.get_parameter("min_line_area").value)
        self.morph_kernel = max(1, int(self.get_parameter("morph_kernel").value))
        if self.morph_kernel % 2 == 0:
            self.morph_kernel += 1

        self.kp = float(self.get_parameter("kp").value)
        self.ki = float(self.get_parameter("ki").value)
        self.kd = float(self.get_parameter("kd").value)
        self.max_yaw_rate = abs(float(self.get_parameter("max_yaw_rate").value))
        self.integral_limit = abs(float(self.get_parameter("integral_limit").value))

        self.base_speed = float(self.get_parameter("base_speed").value)
        self.min_speed = float(self.get_parameter("min_speed").value)
        self.max_speed = float(self.get_parameter("max_speed").value)
        self.speed_reduction = float(self.get_parameter("speed_reduction").value)
        if self.max_speed <= 0.0 or self.min_speed < 0.0 or self.min_speed > self.max_speed:
            raise ValueError("Invalid speed limits.")
        self.base_speed = min(max(self.base_speed, self.min_speed), self.max_speed)

        self.image_timeout = float(self.get_parameter("image_timeout").value)
        self.log_interval = float(self.get_parameter("log_interval").value)
        self.show_debug = bool(self.get_parameter("show_debug").value)

        self.bridge = CvBridge()
        self.pub = self.create_publisher(Twist, self.cmd_topic, 10)

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.sub = self.create_subscription(Image, self.image_topic, self.image_callback, qos)

        self.last_image_wall = None
        self.last_control_wall = None
        self.last_log_wall = 0.0
        self.last_error = 0.0
        self.integral = 0.0
        self.derivative_filtered = 0.0
        self.line_was_visible = False
        self.stopped_for_timeout = False

        # Watchdog: if the camera stops publishing, stop the robot.
        self.watchdog = self.create_timer(0.05, self.watchdog_callback)

        self.get_logger().info(
            f"Line follower ready. image={self.image_topic}, cmd={self.cmd_topic}"
        )
        self.get_logger().info(
            "Detecting yellow centreline in HSV; controller publishes only when a line is detected."
        )

    def publish_stop(self):
        stop_msg = Twist()

        stop_msg.linear.x = 0.0
        stop_msg.linear.y = 0.0
        stop_msg.linear.z = 0.0

        stop_msg.angular.x = 0.0
        stop_msg.angular.y = 0.0
        stop_msg.angular.z = 0.0

        for _ in range(10):
            self.pub.publish(stop_msg)
            time.sleep(0.05)

        self.get_logger().info("STOP: Robot zatrzymany.")


    def image_callback(self, msg: Image):
        now = time.monotonic()
        self.last_image_wall = now
        self.stopped_for_timeout = False

        try:
            # Gazebo bridge publishes RGB images. cv_bridge handles ROS encoding.
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"Could not decode camera image: {exc}")
            self.publish_stop()
            return

        if frame is None or frame.size == 0:
            self.publish_stop()
            return

        height, width = frame.shape[:2]
        y0 = int(height * self.roi_top)
        y1 = max(y0 + 1, int(height * self.roi_bottom))
        roi = frame[y0:y1, :]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower = np.array([self.h_low, self.s_low, self.v_low], dtype=np.uint8)
        upper = np.array([self.h_high, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        k = self.morph_kernel
        if k > 1:
            kernel = np.ones((k, k), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.handle_line_lost(now)
            return

        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        if area < self.min_line_area:
            self.handle_line_lost(now)
            return

        moments = cv2.moments(contour)
        if moments["m00"] <= 1e-6:
            self.handle_line_lost(now)
            return

        line_x = moments["m10"] / moments["m00"]
        image_center_x = width / 2.0

        # Positive error means the line is left of image centre.
        error = (image_center_x - line_x) / max(image_center_x, 1.0)
        error = max(-1.0, min(1.0, error))

        dt = 0.05 if self.last_control_wall is None else now - self.last_control_wall
        dt = min(max(dt, 0.02), 0.5)
        self.last_control_wall = now

        self.integral += error * dt
        self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))

        derivative = (error - self.last_error) / dt
        # Light low-pass filtering to reduce noise from image segmentation.
        self.derivative_filtered = 0.65 * self.derivative_filtered + 0.35 * derivative
        self.last_error = error

        yaw_rate = (
            self.kp * error
            + self.ki * self.integral
            + self.kd * self.derivative_filtered
        )
        yaw_rate = max(-self.max_yaw_rate, min(self.max_yaw_rate, yaw_rate))

        # Reduce speed when the line is far from the image centre.
        speed_factor = max(0.0, 1.0 - self.speed_reduction * abs(error))
        speed = self.base_speed * speed_factor
        speed = max(self.min_speed, min(self.max_speed, speed))

        cmd = Twist()
        cmd.linear.x = float(speed)
        cmd.angular.z = float(yaw_rate)
        self.pub.publish(cmd)

        self.line_was_visible = True
        self.stopped_for_timeout = False

        if now - self.last_log_wall >= self.log_interval:
            self.get_logger().info(
                f"line_x={line_x:.1f}/{width}, error={error:+.3f}, "
                f"v={speed:.2f} m/s, wz={yaw_rate:+.2f} rad/s, area={area:.0f}"
            )
            self.last_log_wall = now

        if self.show_debug:
            debug = frame.copy()
            cv2.rectangle(debug, (0, y0), (width - 1, y1), (255, 0, 0), 1)
            cv2.circle(debug, (int(line_x), int((y0 + y1) / 2)), 7, (0, 0, 255), -1)
            cv2.line(debug, (width // 2, y0), (width // 2, y1), (0, 255, 0), 2)
            cv2.imshow("RoboRacer Line Follower", debug)
            cv2.imshow("Yellow line mask", mask)
            cv2.waitKey(1)

    def handle_line_lost(self, now):
        self.publish_stop()
        self.integral = 0.0
        self.derivative_filtered = 0.0
        if self.line_was_visible:
            self.get_logger().warning("Line lost: stopping robot.")
        self.line_was_visible = False
        if self.show_debug:
            cv2.waitKey(1)

    def watchdog_callback(self):
        if self.last_image_wall is None:
            return
        age = time.monotonic() - self.last_image_wall
        if age > self.image_timeout and not self.stopped_for_timeout:
            self.publish_stop()
            self.stopped_for_timeout = True
            self.get_logger().warning(
                f"No camera frame for {age:.2f} s: watchdog stopped the robot."
            )

    def destroy_node(self):
        try:
            self.publish_stop()
        except Exception:
            pass
        if self.show_debug:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = LineFollower()
        rclpy.spin(node)
    except (ValueError, RuntimeError) as exc:
        print(f"[line_follower] Configuration/runtime error: {exc}")
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            try:
                node.publish_stop()
            except Exception:
                pass
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
