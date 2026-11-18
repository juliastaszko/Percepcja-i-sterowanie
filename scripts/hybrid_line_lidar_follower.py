#!/usr/bin/env python3

import math
import time
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, LaserScan


class HybridLineLidarFollower(Node):
    def __init__(self):
        super().__init__("roboracer_hybrid_line_lidar_follower")

        # Topics
        self.declare_parameter("image_topic", "/camera/color/image_raw")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")

        # Camera segmentation (same starting values as the original follower)
        self.declare_parameter("roi_top", 0.55)
        self.declare_parameter("roi_bottom", 0.96)
        self.declare_parameter("hsv_h_low", 15)
        self.declare_parameter("hsv_h_high", 40)
        self.declare_parameter("hsv_s_low", 90)
        self.declare_parameter("hsv_v_low", 90)
        self.declare_parameter("min_line_area", 35.0)
        self.declare_parameter("morph_kernel", 3)

        # Camera PID and speed
        self.declare_parameter("kp", 1.25)
        self.declare_parameter("ki", 0.0)
        self.declare_parameter("kd", 0.08)
        self.declare_parameter("max_yaw_rate", 1.4)
        self.declare_parameter("integral_limit", 1.0)
        self.declare_parameter("base_speed", 0.22)
        self.declare_parameter("min_speed", 0.08)
        self.declare_parameter("max_speed", 0.30)
        self.declare_parameter("speed_reduction", 0.55)

        # Mode switching / sensor watchdogs
        self.declare_parameter("line_confirm_frames", 3)
        self.declare_parameter("line_lost_frames", 2)
        self.declare_parameter("camera_timeout", 1.0)
        self.declare_parameter("scan_timeout", 1.0)
        self.declare_parameter("control_rate", 20.0)

        # LiDAR parameters (metres/degrees)
        self.declare_parameter("front_angle_deg", 25.0)
        self.declare_parameter("side_angle_deg", 100.0)
        self.declare_parameter("obstacle_distance", 0.65)
        self.declare_parameter("critical_distance", 0.28)
        self.declare_parameter("safety_distance", 0.45)
        self.declare_parameter("gap_min_distance", 0.75)
        self.declare_parameter("lidar_speed", 0.12)
        self.declare_parameter("lidar_max_speed", 0.16)
        self.declare_parameter("lidar_max_yaw_rate", 1.0)
        self.declare_parameter("gap_smoothing_window", 5)
        self.declare_parameter("lidar_turn_gain", 1.8)
        self.declare_parameter("front_slow_distance", 1.2)
        self.declare_parameter("log_interval", 1.0)
        self.declare_parameter("show_debug", False)

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.scan_topic = str(self.get_parameter("scan_topic").value)
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
        if self.max_speed <= 0 or self.min_speed < 0 or self.min_speed > self.max_speed:
            raise ValueError("Invalid camera speed limits.")
        self.base_speed = float(np.clip(self.base_speed, self.min_speed, self.max_speed))

        self.line_confirm_frames = max(1, int(self.get_parameter("line_confirm_frames").value))
        self.line_lost_frames = max(1, int(self.get_parameter("line_lost_frames").value))
        self.camera_timeout = float(self.get_parameter("camera_timeout").value)
        self.scan_timeout = float(self.get_parameter("scan_timeout").value)
        rate = max(1.0, float(self.get_parameter("control_rate").value))

        self.front_angle = math.radians(abs(float(self.get_parameter("front_angle_deg").value)))
        self.side_angle = math.radians(abs(float(self.get_parameter("side_angle_deg").value)))
        self.obstacle_distance = float(self.get_parameter("obstacle_distance").value)
        self.critical_distance = float(self.get_parameter("critical_distance").value)
        self.safety_distance = float(self.get_parameter("safety_distance").value)
        self.gap_min_distance = float(self.get_parameter("gap_min_distance").value)
        self.lidar_speed = float(self.get_parameter("lidar_speed").value)
        self.lidar_max_speed = float(self.get_parameter("lidar_max_speed").value)
        self.lidar_max_yaw_rate = abs(float(self.get_parameter("lidar_max_yaw_rate").value))
        self.gap_window = max(1, int(self.get_parameter("gap_smoothing_window").value))
        self.lidar_turn_gain = float(self.get_parameter("lidar_turn_gain").value)
        self.front_slow_distance = float(self.get_parameter("front_slow_distance").value)
        self.log_interval = float(self.get_parameter("log_interval").value)
        self.show_debug = bool(self.get_parameter("show_debug").value)

        self.bridge = CvBridge()
        self.pub = self.create_publisher(Twist, self.cmd_topic, 10)
        image_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                               reliability=ReliabilityPolicy.RELIABLE)
        scan_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=5,
                              reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, self.image_topic, self.image_callback, image_qos)
        self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, scan_qos)

        self.last_image_wall: Optional[float] = None
        self.last_scan_wall: Optional[float] = None
        self.last_control_wall: Optional[float] = None
        self.last_log_wall = 0.0
        self.line_error: Optional[float] = None
        self.line_detect_count = 0
        self.line_lost_count = 0
        self.scan_ranges = None
        self.scan_angle_min = 0.0
        self.scan_angle_increment = 0.0
        self.scan_range_min = 0.05
        self.scan_range_max = 10.0
        self.mode = "WAITING"
        self.integral = 0.0
        self.last_error = 0.0
        self.derivative_filtered = 0.0
        self.create_timer(1.0 / rate, self.control_loop)

        self.get_logger().info(
            f"Hybrid follower ready: image={self.image_topic}, scan={self.scan_topic}, cmd={self.cmd_topic}"
        )

    def publish_cmd(self, v, wz):
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(wz)
        self.pub.publish(msg)

    def publish_stop(self):
        self.publish_cmd(0.0, 0.0)

    def image_callback(self, msg: Image):
        self.last_image_wall = time.monotonic()
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"Camera conversion failed: {exc}")
            self.register_line_missing()
            return
        if frame is None or frame.size == 0:
            self.register_line_missing()
            return

        height, width = frame.shape[:2]
        y0, y1 = int(height * self.roi_top), int(height * self.roi_bottom)
        roi = frame[y0:max(y0 + 1, y1), :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower = np.array([self.h_low, self.s_low, self.v_low], dtype=np.uint8)
        upper = np.array([self.h_high, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        if self.morph_kernel > 1:
            kernel = np.ones((self.morph_kernel, self.morph_kernel), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.register_line_missing()
            return
        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        moments = cv2.moments(contour)
        if area < self.min_line_area or moments["m00"] <= 1e-6:
            self.register_line_missing()
            return

        line_x = moments["m10"] / moments["m00"]
        self.line_error = float(np.clip((width / 2.0 - line_x) / max(width / 2.0, 1.0), -1.0, 1.0))
        self.line_detect_count += 1
        self.line_lost_count = 0

        if self.show_debug:
            debug = frame.copy()
            cv2.rectangle(debug, (0, y0), (width - 1, y1), (255, 0, 0), 1)
            cv2.circle(debug, (int(line_x), int((y0 + y1) / 2)), 7, (0, 0, 255), -1)
            cv2.imshow("Hybrid line follower", debug)
            cv2.imshow("Yellow line mask", mask)
            cv2.waitKey(1)

    def register_line_missing(self):
        self.line_lost_count += 1
        self.line_detect_count = 0
        if self.line_lost_count >= self.line_lost_frames:
            self.line_error = None

    def scan_callback(self, msg: LaserScan):
        self.last_scan_wall = time.monotonic()
        self.scan_ranges = np.asarray(msg.ranges, dtype=np.float32)
        self.scan_angle_min = float(msg.angle_min)
        self.scan_angle_increment = float(msg.angle_increment)
        self.scan_range_min = float(msg.range_min)
        self.scan_range_max = float(msg.range_max)

    def scan_valid(self):
        return (self.scan_ranges is not None and self.last_scan_wall is not None
                and time.monotonic() - self.last_scan_wall <= self.scan_timeout
                and len(self.scan_ranges) > 0 and abs(self.scan_angle_increment) > 1e-9)

    def get_scan(self):
        n = len(self.scan_ranges)
        angles = self.scan_angle_min + np.arange(n, dtype=np.float32) * self.scan_angle_increment
        ranges = self.scan_ranges.copy()
        ranges[~np.isfinite(ranges)] = self.scan_range_max
        ranges = np.clip(ranges, self.scan_range_min, self.scan_range_max)
        return ranges, angles

    def sector_min(self, ranges, angles, center, half_width):
        delta = np.arctan2(np.sin(angles - center), np.cos(angles - center))
        values = ranges[np.abs(delta) <= half_width]
        return float(np.min(values)) if values.size else self.scan_range_max

    def front_distance(self):
        ranges, angles = self.get_scan()
        return self.sector_min(ranges, angles, 0.0, self.front_angle)

    def lidar_command(self):
        """Simple Follow-the-Gap reactive obstacle avoidance."""
        ranges, angles = self.get_scan()
        forward = np.abs(np.arctan2(np.sin(angles), np.cos(angles))) <= self.side_angle
        indices = np.where(forward)[0]
        if indices.size < 5:
            return 0.0, 0.0, "insufficient_forward_scan"

        local = ranges[indices].copy()
        closest_i = int(np.argmin(local))
        closest_r = float(local[closest_i])
        bubble_angle = math.atan2(self.safety_distance, max(closest_r, 0.05))
        bubble_n = max(1, int(bubble_angle / max(abs(self.scan_angle_increment), 1e-4)))
        local[max(0, closest_i - bubble_n):min(len(local), closest_i + bubble_n + 1)] = 0.0

        if self.gap_window > 1 and len(local) >= self.gap_window:
            kernel = np.ones(self.gap_window, dtype=np.float32) / self.gap_window
            smooth = np.convolve(local, kernel, mode="same")
        else:
            smooth = local

        free = smooth >= self.gap_min_distance
        intervals = []
        start = None
        for i, ok in enumerate(free):
            if ok and start is None:
                start = i
            elif not ok and start is not None:
                intervals.append((start, i - 1))
                start = None
        if start is not None:
            intervals.append((start, len(free) - 1))

        best_angle, best_score = None, -1.0
        for a, b in intervals:
            if b - a + 1 < 2:
                continue
            mid = (a + b) // 2
            direction = float(angles[indices[mid]])
            clearance = float(np.max(smooth[a:b + 1]))
            score = (b - a + 1) * min(clearance, 4.0) / (1.0 + 0.8 * abs(direction))
            if score > best_score:
                best_score, best_angle = score, direction

        front = self.front_distance()
        if front < self.critical_distance:
            return 0.0, 0.0, f"critical_stop front={front:.2f}m"

        if best_angle is None:
            left = self.sector_min(ranges, angles, math.pi / 2, math.radians(35))
            right = self.sector_min(ranges, angles, -math.pi / 2, math.radians(35))
            best_angle = math.radians(45) if left >= right else -math.radians(45)

        yaw = float(np.clip(self.lidar_turn_gain * best_angle,
                            -self.lidar_max_yaw_rate, self.lidar_max_yaw_rate))
        turn_factor = max(0.35, 1.0 - abs(best_angle) / max(self.side_angle, 0.01) * 0.65)
        distance_factor = float(np.clip(
            (front - self.critical_distance) /
            max(self.front_slow_distance - self.critical_distance, 0.01), 0.25, 1.0
        ))
        speed = min(self.lidar_speed, self.lidar_max_speed) * turn_factor * distance_factor
        if front < self.obstacle_distance:
            speed = min(speed, self.lidar_speed * 0.65)
        return float(speed), yaw, f"gap={math.degrees(best_angle):.1f}deg front={front:.2f}m"

    def camera_command(self, now):
        error = float(self.line_error)
        dt = 0.05 if self.last_control_wall is None else now - self.last_control_wall
        dt = float(np.clip(dt, 0.02, 0.5))
        self.last_control_wall = now
        self.integral = float(np.clip(self.integral + error * dt,
                                      -self.integral_limit, self.integral_limit))
        derivative = (error - self.last_error) / dt
        self.derivative_filtered = 0.65 * self.derivative_filtered + 0.35 * derivative
        self.last_error = error
        yaw = self.kp * error + self.ki * self.integral + self.kd * self.derivative_filtered
        yaw = float(np.clip(yaw, -self.max_yaw_rate, self.max_yaw_rate))
        factor = max(0.0, 1.0 - self.speed_reduction * abs(error))
        speed = float(np.clip(self.base_speed * factor, self.min_speed, self.max_speed))
        return speed, yaw

    def control_loop(self):
        now = time.monotonic()
        # Fail safe: without a fresh LiDAR scan there is no valid obstacle safety layer.
        if not self.scan_valid():
            self.mode = "WAITING_FOR_LIDAR"
            self.publish_stop()
            return

        camera_fresh = (self.last_image_wall is not None
                        and now - self.last_image_wall <= self.camera_timeout)
        line_ok = (camera_fresh and self.line_error is not None
                   and self.line_detect_count >= self.line_confirm_frames)
        front = self.front_distance()
        obstacle = front < self.obstacle_distance

        # LiDAR overrides camera when an obstacle is close; otherwise use camera
        # only after the line has been seen in several consecutive frames.
        if line_ok and not obstacle:
            self.mode = "CAMERA_LINE"
            v, wz = self.camera_command(now)
            detail = f"line_error={self.line_error:+.3f}"
        else:
            self.mode = "LIDAR_AVOID" if obstacle else "LIDAR_SEARCH"
            v, wz, detail = self.lidar_command()

        self.publish_cmd(v, wz)
        if now - self.last_log_wall >= self.log_interval:
            self.get_logger().info(
                f"mode={self.mode} v={v:.2f} wz={wz:+.2f} front={front:.2f}m {detail}"
            )
            self.last_log_wall = now

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
        node = HybridLineLidarFollower()
        rclpy.spin(node)
    except (ValueError, RuntimeError) as exc:
        print(f"[hybrid_line_lidar_follower] Error: {exc}")
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
