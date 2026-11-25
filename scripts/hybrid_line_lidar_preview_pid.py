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


class HybridPreviewPIDFollower(Node):
    def __init__(self):
        super().__init__("roboracer_hybrid_preview_pid_follower")

        # =====================================================
        # TOPICS
        # =====================================================

        self.declare_parameter("image_topic", "/camera/color/image_raw")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")

        # =====================================================
        # CAMERA SEGMENTATION
        # =====================================================

        self.declare_parameter("roi_top", 0.55)
        self.declare_parameter("roi_bottom", 0.96)

        self.declare_parameter("hsv_h_low", 15)
        self.declare_parameter("hsv_h_high", 40)

        self.declare_parameter("hsv_s_low", 90)
        self.declare_parameter("hsv_v_low", 90)

        self.declare_parameter("min_line_area", 35.0)
        self.declare_parameter("morph_kernel", 3)

        # =====================================================
        # CAMERA PID AND SPEED
        # =====================================================

        self.declare_parameter("kp", 1.25)
        self.declare_parameter("ki", 0.0)
        self.declare_parameter("kd", 0.08)

        self.declare_parameter("max_yaw_rate", 1.4)
        self.declare_parameter("integral_limit", 1.0)

        # =====================================================
        # HYBRID PREVIEW PID PARAMETERS
        # =====================================================

        # Number of horizontal image bands used to estimate line shape.
        self.declare_parameter("preview_bands", 9)
        self.declare_parameter("preview_min_band_pixels", 3)

        # Preview controller gains.
        self.declare_parameter("preview_kp", 0.85)
        self.declare_parameter("preview_kd", 0.035)
        self.declare_parameter("preview_heading_gain", 0.55)
        self.declare_parameter("preview_curvature_gain", 0.18)

        # Distance (in normalized image coordinates) from the bottom
        # of the ROI at which the preview lateral error is evaluated.
        self.declare_parameter("preview_error_position", 0.18)

        # Speed adaptation based on estimated heading and curvature.
        self.declare_parameter("preview_heading_slowdown", 0.30)
        self.declare_parameter("preview_curvature_slowdown", 0.22)

        self.declare_parameter("base_speed", 0.50)
        self.declare_parameter("min_speed", 0.08)
        self.declare_parameter("max_speed", 0.50)

        self.declare_parameter("speed_reduction", 0.55)

        # =====================================================
        # MODE SWITCHING / SENSOR WATCHDOGS
        # =====================================================

        self.declare_parameter("line_confirm_frames", 3)
        self.declare_parameter("line_lost_frames", 2)

        self.declare_parameter("camera_timeout", 1.0)
        self.declare_parameter("scan_timeout", 1.0)

        self.declare_parameter("control_rate", 20.0)

        # =====================================================
        # LIDAR PARAMETERS
        # =====================================================

        self.declare_parameter("front_angle_deg", 30.0)
        self.declare_parameter("side_angle_deg", 85.0)

        self.declare_parameter("obstacle_distance", 0.85)
        self.declare_parameter("critical_distance", 0.30)

        self.declare_parameter("safety_distance", 0.45)

        self.declare_parameter("vehicle_width", 0.31)
        self.declare_parameter("vehicle_side_margin", 0.12)

        self.declare_parameter("gap_min_distance", 0.75)

        self.declare_parameter("gap_direction_memory", 2.5)
        self.declare_parameter("avoidance_hold_time", 1.5)

        self.declare_parameter("lidar_speed", 0.10)
        self.declare_parameter("lidar_max_speed", 0.12)

        self.declare_parameter("lidar_max_yaw_rate", 0.70)

        self.declare_parameter("gap_smoothing_window", 5)

        self.declare_parameter("lidar_turn_gain", 1.4)

        self.declare_parameter("front_slow_distance", 1.2)

        self.declare_parameter("log_interval", 1.0)
        self.declare_parameter("show_debug", False)

        # =====================================================
        # READ PARAMETERS
        # =====================================================

        self.image_topic = str(
            self.get_parameter("image_topic").value
        )

        self.scan_topic = str(
            self.get_parameter("scan_topic").value
        )

        self.cmd_topic = str(
            self.get_parameter("cmd_vel_topic").value
        )

        # Camera ROI

        self.roi_top = float(
            self.get_parameter("roi_top").value
        )

        self.roi_bottom = float(
            self.get_parameter("roi_bottom").value
        )

        if not (0.0 <= self.roi_top < self.roi_bottom <= 1.0):
            raise ValueError("Require 0 <= roi_top < roi_bottom <= 1.")

        # HSV

        self.h_low = int(
            self.get_parameter("hsv_h_low").value
        )

        self.h_high = int(
            self.get_parameter("hsv_h_high").value
        )

        self.s_low = int(
            self.get_parameter("hsv_s_low").value
        )

        self.v_low = int(
            self.get_parameter("hsv_v_low").value
        )

        self.min_line_area = float(
            self.get_parameter("min_line_area").value
        )

        self.morph_kernel = max(
            1,
            int(self.get_parameter("morph_kernel").value)
        )

        if self.morph_kernel % 2 == 0:
            self.morph_kernel += 1

        # PID

        self.kp = float(self.get_parameter("kp").value)
        self.ki = float(self.get_parameter("ki").value)
        self.kd = float(self.get_parameter("kd").value)

        self.max_yaw_rate = abs(
            float(self.get_parameter("max_yaw_rate").value)
        )

        self.integral_limit = abs(
            float(self.get_parameter("integral_limit").value)
        )

        self.preview_bands = max(
            5, int(self.get_parameter("preview_bands").value)
        )
        self.preview_min_band_pixels = max(
            1, int(self.get_parameter("preview_min_band_pixels").value)
        )

        self.preview_kp = float(self.get_parameter("preview_kp").value)
        self.preview_kd = float(self.get_parameter("preview_kd").value)
        self.preview_heading_gain = float(
            self.get_parameter("preview_heading_gain").value
        )
        self.preview_curvature_gain = float(
            self.get_parameter("preview_curvature_gain").value
        )
        self.preview_error_position = float(
            np.clip(self.get_parameter("preview_error_position").value, 0.0, 0.8)
        )
        self.preview_heading_slowdown = max(
            0.0, float(self.get_parameter("preview_heading_slowdown").value)
        )
        self.preview_curvature_slowdown = max(
            0.0, float(self.get_parameter("preview_curvature_slowdown").value)
        )

        self.base_speed = float(
            self.get_parameter("base_speed").value
        )

        self.min_speed = float(
            self.get_parameter("min_speed").value
        )

        self.max_speed = float(
            self.get_parameter("max_speed").value
        )

        self.speed_reduction = float(
            self.get_parameter("speed_reduction").value
        )

        if (
            self.max_speed <= 0
            or self.min_speed < 0
            or self.min_speed > self.max_speed
        ):
            raise ValueError("Invalid camera speed limits.")

        self.base_speed = float(
            np.clip(
                self.base_speed,
                self.min_speed,
                self.max_speed
            )
        )

        # Watchdogs

        self.line_confirm_frames = max(
            1,
            int(self.get_parameter("line_confirm_frames").value)
        )

        self.line_lost_frames = max(
            1,
            int(self.get_parameter("line_lost_frames").value)
        )

        self.camera_timeout = float(
            self.get_parameter("camera_timeout").value
        )

        self.scan_timeout = float(
            self.get_parameter("scan_timeout").value
        )

        rate = max(
            1.0,
            float(self.get_parameter("control_rate").value)
        )

        # LiDAR

        self.front_angle = math.radians(
            abs(float(self.get_parameter("front_angle_deg").value))
        )

        self.side_angle = math.radians(
            abs(float(self.get_parameter("side_angle_deg").value))
        )

        self.obstacle_distance = float(
            self.get_parameter("obstacle_distance").value
        )

        self.critical_distance = float(
            self.get_parameter("critical_distance").value
        )

        self.safety_distance = float(
            self.get_parameter("safety_distance").value
        )

        self.vehicle_width = float(
            self.get_parameter("vehicle_width").value
        )

        self.vehicle_side_margin = float(
            self.get_parameter("vehicle_side_margin").value
        )

        self.required_gap_width = (
            self.vehicle_width
            + 2.0 * self.vehicle_side_margin
        )

        self.gap_min_distance = float(
            self.get_parameter("gap_min_distance").value
        )

        self.gap_direction_memory = float(
            self.get_parameter("gap_direction_memory").value
        )

        self.avoidance_hold_time = float(
            self.get_parameter("avoidance_hold_time").value
        )

        self.lidar_speed = float(
            self.get_parameter("lidar_speed").value
        )

        self.lidar_max_speed = float(
            self.get_parameter("lidar_max_speed").value
        )

        self.lidar_max_yaw_rate = abs(
            float(self.get_parameter("lidar_max_yaw_rate").value)
        )

        self.gap_window = max(
            1,
            int(self.get_parameter("gap_smoothing_window").value)
        )

        self.lidar_turn_gain = float(
            self.get_parameter("lidar_turn_gain").value
        )

        self.front_slow_distance = float(
            self.get_parameter("front_slow_distance").value
        )

        self.log_interval = float(
            self.get_parameter("log_interval").value
        )

        self.show_debug = bool(
            self.get_parameter("show_debug").value
        )

        # =====================================================
        # ROS2 INITIALIZATION
        # =====================================================

        self.bridge = CvBridge()

        self.pub = self.create_publisher(
            Twist,
            self.cmd_topic,
            10
        )

        from std_msgs.msg import String
        self.mode_pub = self.create_publisher(
            String,
            "/control_mode",
            10
        )

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE
        )

        scan_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )

        self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            image_qos
        )

        self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.scan_callback,
            scan_qos
        )

        # =====================================================
        # INTERNAL VARIABLES
        # =====================================================

        self.last_image_wall: Optional[float] = None
        self.last_scan_wall: Optional[float] = None
        self.last_control_wall: Optional[float] = None

        self.last_log_wall = 0.0

        self.line_error: Optional[float] = None
        self.preview_error: Optional[float] = None
        self.preview_heading = 0.0
        self.preview_curvature = 0.0
        self.preview_fit_points = 0

        self.line_detect_count = 0
        self.line_lost_count = 0

        self.scan_ranges = None

        self.scan_angle_min = 0.0
        self.scan_angle_increment = 0.0

        self.scan_range_min = 0.05
        self.scan_range_max = 10.0

        self.mode = "WAITING"

        self.last_gap_angle = None
        self.last_gap_time = 0.0

        self.avoidance_until = 0.0

        self.integral = 0.0
        self.last_error = 0.0

        self.derivative_filtered = 0.0

        self.create_timer(
            1.0 / rate,
            self.control_loop
        )

        self.get_logger().info(
            f"Hybrid follower ready: "
            f"image={self.image_topic}, "
            f"scan={self.scan_topic}, "
            f"cmd={self.cmd_topic}"
        )

    # =========================================================
    # PUBLISHING
    # =========================================================

    def publish_cmd(self, v, wz):

        msg = Twist()

        msg.linear.x = float(v)
        msg.angular.z = float(wz)

        self.pub.publish(msg)

    def publish_stop(self):

        self.publish_cmd(0.0, 0.0)

    # =========================================================
    # CAMERA CALLBACK
    # =========================================================

    def image_callback(self, msg: Image):

        self.last_image_wall = time.monotonic()

        try:

            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8"
            )

        except Exception as exc:

            self.get_logger().error(
                f"Camera conversion failed: {exc}"
            )

            self.register_line_missing()
            return

        if frame is None or frame.size == 0:

            self.register_line_missing()
            return

        height, width = frame.shape[:2]

        y0 = int(height * self.roi_top)
        y1 = int(height * self.roi_bottom)

        roi = frame[y0:max(y0 + 1, y1), :]

        hsv = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2HSV
        )

        lower = np.array(
            [self.h_low, self.s_low, self.v_low],
            dtype=np.uint8
        )

        upper = np.array(
            [self.h_high, 255, 255],
            dtype=np.uint8
        )

        mask = cv2.inRange(
            hsv,
            lower,
            upper
        )

        if self.morph_kernel > 1:

            kernel = np.ones(
                (self.morph_kernel, self.morph_kernel),
                np.uint8
            )

            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_OPEN,
                kernel
            )

            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                kernel
            )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:

            self.register_line_missing()
            return

        contour = max(
            contours,
            key=cv2.contourArea
        )

        area = float(
            cv2.contourArea(contour)
        )

        moments = cv2.moments(contour)

        if area < self.min_line_area or moments["m00"] <= 1e-6:

            self.register_line_missing()
            return

        line_x = moments["m10"] / moments["m00"]

        # Keep the original centroid error as a fallback if the line is too
        # fragmented to produce enough preview samples.
        centroid_error = float(
            np.clip(
                (width / 2.0 - line_x)
                / max(width / 2.0, 1.0),
                -1.0,
                1.0
            )
        )

        preview = self.estimate_line_preview(
            contour,
            roi.shape[0],
            width
        )

        if preview is None:
            self.preview_error = centroid_error
            self.preview_heading = 0.0
            self.preview_curvature = 0.0
            self.preview_fit_points = 0
            self.line_error = centroid_error
            preview_points = []
        else:
            (
                self.preview_error,
                self.preview_heading,
                self.preview_curvature,
                self.preview_fit_points,
                preview_points
            ) = preview
            self.line_error = self.preview_error

        self.line_detect_count += 1
        self.line_lost_count = 0

        if self.show_debug:

            debug = frame.copy()

            cv2.rectangle(
                debug,
                (0, y0),
                (width - 1, y1),
                (255, 0, 0),
                1
            )

            cv2.circle(
                debug,
                (int(line_x), int((y0 + y1) / 2)),
                7,
                (0, 0, 255),
                -1
            )

            # Draw sampled preview points and fitted line.
            for s, x_norm in preview_points:
                px = int(width / 2.0 - x_norm * width / 2.0)
                py = int(y0 + (1.0 - s) * (roi.shape[0] - 1))
                cv2.circle(debug, (px, py), 4, (0, 255, 0), -1)

            cv2.putText(
                debug,
                f"e={self.preview_error:+.2f} "
                f"heading={self.preview_heading:+.2f} "
                f"curv={self.preview_curvature:+.2f}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

            cv2.imshow("Hybrid Preview PID", debug)
            cv2.imshow("Yellow line mask", mask)
            cv2.waitKey(1)

    # =========================================================
    # PREVIEW GEOMETRY ESTIMATION
    # =========================================================

    def estimate_line_preview(self, contour, roi_height, image_width):
        """
        Estimate the yellow line geometry from several horizontal bands.

        Coordinate convention:
          s = 0: bottom/near part of ROI
          s = 1: top/far part of ROI
          x_norm > 0: line lies to the right of image center

        The signs of heading and curvature are converted so that positive
        values correspond to a left turn, matching the existing PID sign.
        """
        contour_mask = np.zeros((roi_height, image_width), dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)

        band_centers = np.linspace(
            int(roi_height * 0.08),
            int(roi_height * 0.94),
            self.preview_bands
        )

        points_s = []
        points_x = []

        band_half_height = max(1, int(roi_height * 0.035))

        for row in band_centers:
            y = int(round(row))
            y0 = max(0, y - band_half_height)
            y1 = min(roi_height, y + band_half_height + 1)

            ys, xs = np.where(contour_mask[y0:y1, :] > 0)

            if xs.size < self.preview_min_band_pixels:
                continue

            # Median is less sensitive than a mean to isolated contour pixels.
            x_center = float(np.median(xs))
            s = float((roi_height - 1 - y) / max(roi_height - 1, 1))
            x_norm = float((image_width / 2.0 - x_center) / max(image_width / 2.0, 1.0))

            points_s.append(s)
            points_x.append(x_norm)

        if len(points_s) < 4:
            return None

        s_values = np.asarray(points_s, dtype=np.float64)
        x_values = np.asarray(points_x, dtype=np.float64)

        try:
            # x(s) = a*s^2 + b*s + c
            a, b, c = np.polyfit(s_values, x_values, 2)
        except (ValueError, np.linalg.LinAlgError):
            return None

        s_eval = self.preview_error_position
        x_preview = float(a * s_eval * s_eval + b * s_eval + c)

        # Positive heading/curvature means the line bends toward image-left.
        dx_ds = float(2.0 * a * s_eval + b)
        heading = -math.atan(dx_ds)

        second_derivative = float(2.0 * a)
        curvature = -second_derivative / (
            (1.0 + dx_ds * dx_ds) ** 1.5
        )

        # Limit noisy image-space estimates before they enter the controller.
        x_preview = float(np.clip(x_preview, -1.0, 1.0))
        heading = float(np.clip(heading, -1.2, 1.2))
        curvature = float(np.clip(curvature, -4.0, 4.0))

        return x_preview, heading, curvature, len(points_s), list(zip(points_s, points_x))

    # =========================================================
    # LINE LOST
    # =========================================================

    def register_line_missing(self):

        self.line_lost_count += 1
        self.line_detect_count = 0

        if self.line_lost_count >= self.line_lost_frames:

            self.line_error = None
            self.preview_error = None
            self.preview_heading = 0.0
            self.preview_curvature = 0.0
            self.preview_fit_points = 0

    # =========================================================
    # LIDAR CALLBACK
    # =========================================================

    def scan_callback(self, msg: LaserScan):

        self.last_scan_wall = time.monotonic()

        self.scan_ranges = np.asarray(
            msg.ranges,
            dtype=np.float32
        )

        self.scan_angle_min = float(msg.angle_min)

        self.scan_angle_increment = float(
            msg.angle_increment
        )

        self.scan_range_min = float(msg.range_min)
        self.scan_range_max = float(msg.range_max)

    # =========================================================
    # LIDAR VALIDATION
    # =========================================================

    def scan_valid(self):

        return (
            self.scan_ranges is not None
            and self.last_scan_wall is not None
            and time.monotonic() - self.last_scan_wall <= self.scan_timeout
            and len(self.scan_ranges) > 0
            and abs(self.scan_angle_increment) > 1e-9
        )

    # =========================================================
    # GET SCAN
    # =========================================================

    def get_scan(self):

        n = len(self.scan_ranges)

        angles = (
            self.scan_angle_min
            + np.arange(n, dtype=np.float32)
            * self.scan_angle_increment
        )

        ranges = self.scan_ranges.copy()

        ranges[~np.isfinite(ranges)] = self.scan_range_max

        ranges = np.clip(
            ranges,
            self.scan_range_min,
            self.scan_range_max
        )

        return ranges, angles

    # =========================================================
    # SECTOR MINIMUM
    # =========================================================

    def sector_min(
        self,
        ranges,
        angles,
        center,
        half_width
    ):

        delta = np.arctan2(
            np.sin(angles - center),
            np.cos(angles - center)
        )

        values = ranges[
            np.abs(delta) <= half_width
        ]

        return (
            float(np.min(values))
            if values.size
            else self.scan_range_max
        )

    # =========================================================
    # FRONT DISTANCE
    # =========================================================

    def front_distance(self):

        ranges, angles = self.get_scan()

        return self.sector_min(
            ranges,
            angles,
            0.0,
            self.front_angle
        )

    # =========================================================
    # LIDAR COMMAND - IMPROVED FOLLOW THE GAP
    # =========================================================

    def lidar_command(self):

        """
        Conservative Follow-the-Gap.

        If no sufficiently wide gap is found,
        the robot stops instead of choosing
        an arbitrary direction.
        """

        now = time.monotonic()

        ranges, angles = self.get_scan()

        # -----------------------------------------------------
        # FORWARD FIELD OF VIEW
        # -----------------------------------------------------

        forward = (
            np.abs(
                np.arctan2(
                    np.sin(angles),
                    np.cos(angles)
                )
            )
            <= self.side_angle
        )

        indices = np.where(forward)[0]

        if indices.size < 5:

            return (
                0.0,
                0.0,
                "insufficient_forward_scan"
            )

        local = ranges[indices].copy()

        local_angles = angles[indices]

        # -----------------------------------------------------
        # FIND CLOSEST OBSTACLE
        # -----------------------------------------------------

        closest_i = int(
            np.argmin(local)
        )

        closest_r = float(
            local[closest_i]
        )

        # -----------------------------------------------------
        # OBSTACLE INFLATION
        # -----------------------------------------------------

        inflation = max(
            self.safety_distance,
            0.5 * self.vehicle_width
            + self.vehicle_side_margin
        )

        bubble_angle = math.atan2(
            inflation,
            max(closest_r, 0.05)
        )

        bubble_n = max(
            1,
            int(
                bubble_angle
                / max(
                    abs(self.scan_angle_increment),
                    1e-4
                )
            )
        )

        bubble_start = max(
            0,
            closest_i - bubble_n
        )

        bubble_end = min(
            len(local),
            closest_i + bubble_n + 1
        )

        local[bubble_start:bubble_end] = 0.0

        # -----------------------------------------------------
        # SMOOTHING
        # -----------------------------------------------------

        if (
            self.gap_window > 1
            and len(local) >= self.gap_window
        ):

            kernel = (
                np.ones(
                    self.gap_window,
                    dtype=np.float32
                )
                / self.gap_window
            )

            smooth = np.convolve(
                local,
                kernel,
                mode="same"
            )

        else:

            smooth = local

        # -----------------------------------------------------
        # FREE SPACE DETECTION
        # -----------------------------------------------------

        free = (
            smooth >= self.gap_min_distance
        )

        intervals = []

        gap_start = None

        for i, is_free in enumerate(free):

            if is_free and gap_start is None:

                gap_start = i

            elif not is_free and gap_start is not None:

                intervals.append(
                    (gap_start, i - 1)
                )

                gap_start = None

        if gap_start is not None:

            intervals.append(
                (gap_start, len(free) - 1)
            )

        # -----------------------------------------------------
        # FRONT SAFETY CHECK
        # -----------------------------------------------------

        front = self.front_distance()

        if front <= self.critical_distance:

            return (
                0.0,
                0.0,
                f"critical_stop front={front:.2f}m"
            )

        # -----------------------------------------------------
        # FIND BEST GAP
        # -----------------------------------------------------

        best_angle = None

        best_score = -1.0

        best_width = 0.0

        for a, b in intervals:

            if b - a + 1 < 3:
                continue

            mid = (a + b) // 2

            direction = float(
                local_angles[mid]
            )

            # Reject directions requiring excessively sharp turns.

            if abs(direction) > math.radians(60.0):
                continue

            angular_width = abs(
                float(
                    local_angles[b]
                    - local_angles[a]
                )
            )

            clearance = float(
                np.percentile(
                    smooth[a:b + 1],
                    25
                )
            )

            # Estimate physical width of the gap.

            physical_width = (
                2.0
                * max(clearance, 0.0)
                * math.tan(
                    min(angular_width / 2.0, 1.2)
                )
            )

            # Check whether the robot can fit.

            if physical_width < self.required_gap_width:
                continue

            # Prefer wider and clearer gaps closer to the front.

            direction_penalty = (
                1.0 + 2.5 * abs(direction)
            )

            score = (
                (b - a + 1)
                * min(clearance, 4.0)
                / direction_penalty
            )

            # Direction memory.

            if (
                self.last_gap_angle is not None
                and now - self.last_gap_time
                <= self.gap_direction_memory
                and direction * self.last_gap_angle < 0.0
            ):

                score *= 0.35

            if score > best_score:

                best_score = score

                best_angle = direction

                best_width = physical_width

        # -----------------------------------------------------
        # NO SAFE GAP - STOP
        # -----------------------------------------------------

        if best_angle is None:

            return (
                0.0,
                0.0,
                (
                    f"no_safe_gap "
                    f"front={front:.2f}m "
                    f"required={self.required_gap_width:.2f}m"
                )
            )

        # -----------------------------------------------------
        # SAVE DIRECTION
        # -----------------------------------------------------

        self.last_gap_angle = best_angle
        self.last_gap_time = now

        # -----------------------------------------------------
        # STEERING
        # -----------------------------------------------------

        yaw = float(
            np.clip(
                self.lidar_turn_gain * best_angle,
                -self.lidar_max_yaw_rate,
                self.lidar_max_yaw_rate
            )
        )

        # -----------------------------------------------------
        # SPEED CONTROL
        # -----------------------------------------------------

        turn_factor = max(
            0.30,
            1.0
            - abs(best_angle)
            / max(self.side_angle, 0.01)
            * 0.65
        )

        distance_factor = float(
            np.clip(
                (
                    front - self.critical_distance
                )
                / max(
                    self.front_slow_distance
                    - self.critical_distance,
                    0.01
                ),
                0.25,
                1.0
            )
        )

        speed = (
            min(
                self.lidar_speed,
                self.lidar_max_speed
            )
            * turn_factor
            * distance_factor
        )

        # Additional speed limitation near obstacles.

        if front < self.obstacle_distance:

            speed = min(
                speed,
                0.05
            )

        return (
            float(speed),
            yaw,
            (
                f"gap={math.degrees(best_angle):.1f}deg "
                f"gap_width={best_width:.2f}m "
                f"front={front:.2f}m"
            )
        )

    # =========================================================
    # CAMERA COMMAND - HYBRID PREVIEW PID
    # =========================================================

    def camera_command(self, now):

        error = float(
            self.preview_error
            if self.preview_error is not None
            else self.line_error
        )

        dt = (
            0.05
            if self.last_control_wall is None
            else now - self.last_control_wall
        )
        dt = float(np.clip(dt, 0.02, 0.5))
        self.last_control_wall = now

        derivative = (error - self.last_error) / dt
        self.derivative_filtered = (
            0.65 * self.derivative_filtered
            + 0.35 * derivative
        )
        self.last_error = error

        self.integral = float(
            np.clip(
                self.integral + error * dt,
                -self.integral_limit,
                self.integral_limit
            )
        )

        yaw = (
            self.preview_kp * error
            + self.ki * self.integral
            + self.preview_kd * self.derivative_filtered
            + self.preview_heading_gain * self.preview_heading
            + self.preview_curvature_gain * self.preview_curvature
        )
        yaw = float(np.clip(yaw, -self.max_yaw_rate, self.max_yaw_rate))

        slowdown = (
            self.preview_heading_slowdown * abs(self.preview_heading)
            + self.preview_curvature_slowdown * abs(self.preview_curvature)
        )
        factor = max(
            0.0,
            1.0 - self.speed_reduction * abs(error) - slowdown
        )

        speed = float(
            np.clip(
                self.base_speed * factor,
                self.min_speed,
                self.max_speed
            )
        )

        return speed, yaw

    # =========================================================
    # MAIN CONTROL LOOP
    # =========================================================

    def control_loop(self):

        now = time.monotonic()

        # -----------------------------------------------------
        # LIDAR WATCHDOG
        # -----------------------------------------------------

        if not self.scan_valid():

            self.mode = "WAITING_FOR_LIDAR"

            self.publish_stop()

            return

        # -----------------------------------------------------
        # CAMERA STATUS
        # -----------------------------------------------------

        camera_fresh = (
            self.last_image_wall is not None
            and now - self.last_image_wall <= self.camera_timeout
        )

        line_ok = (
            camera_fresh
            and self.line_error is not None
            and self.line_detect_count >= self.line_confirm_frames
        )

        # -----------------------------------------------------
        # OBSTACLE DETECTION
        # -----------------------------------------------------

        front = self.front_distance()

        obstacle = (
            front < self.obstacle_distance
        )

        if obstacle:

            self.avoidance_until = (
                now + self.avoidance_hold_time
            )

        avoidance_active = (
            obstacle
            or now < self.avoidance_until
        )

        # -----------------------------------------------------
        # MODE SELECTION
        # -----------------------------------------------------

        if line_ok and not avoidance_active:

            self.mode = "CAMERA_LINE"

            v, wz = self.camera_command(now)

            detail = (
                f"preview_error={self.preview_error:+.3f} "
                f"heading={self.preview_heading:+.3f}rad "
                f"curvature={self.preview_curvature:+.3f} "
                f"fit_points={self.preview_fit_points}"
            )

        else:

            if avoidance_active:

                self.mode = "LIDAR_AVOID"

            else:

                self.mode = "LIDAR_SEARCH"

            v, wz, detail = self.lidar_command()

        # -----------------------------------------------------
        # PUBLISH COMMAND AND CONTROL MODE
        # -----------------------------------------------------

        self.publish_cmd(v, wz)

        from std_msgs.msg import String
        mode_msg = String()
        mode_msg.data = "CAMERA" if self.mode == "CAMERA_LINE" else "LIDAR"
        self.mode_pub.publish(mode_msg)

        # -----------------------------------------------------
        # LOGGING
        # -----------------------------------------------------

        if now - self.last_log_wall >= self.log_interval:

            self.get_logger().info(
                f"mode={self.mode} "
                f"v={v:.2f} "
                f"wz={wz:+.2f} "
                f"front={front:.2f}m "
                f"{detail}"
            )

            self.last_log_wall = now

    # =========================================================
    # DESTROY NODE
    # =========================================================

    def destroy_node(self):

        try:

            self.publish_stop()

        except Exception:

            pass

        if self.show_debug:

            cv2.destroyAllWindows()

        super().destroy_node()


# =============================================================
# MAIN
# =============================================================

def main(args=None):

    rclpy.init(args=args)

    node = None

    try:

        node = HybridPreviewPIDFollower()

        rclpy.spin(node)

    except (ValueError, RuntimeError) as exc:

        print(
            f"[hybrid_preview_pid_follower] Error: {exc}"
        )

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