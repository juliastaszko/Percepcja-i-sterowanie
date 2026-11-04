#!/usr/bin/env python3

import csv
import math
from pathlib import Path

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist


def yaw_from_quaternion(q):
    return math.atan2(
        2.0 * (q.w*q.z + q.x*q.y),
        1.0 - 2.0 * (q.y*q.y + q.z*q.z)
    )


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def load_path(filename):

    path = Path(filename).expanduser()

    if not path.is_file():
        raise FileNotFoundError(
            f"CSV trajectory not found: {path}"
        )

    with path.open(
        newline="",
        encoding="utf-8-sig"
    ) as f:

        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise ValueError("CSV requires a header row.")

        cols = {
            c.strip().lower(): c
            for c in reader.fieldnames
        }

        xc = cols.get("x")
        yc = cols.get("y")

        if xc is None or yc is None:
            raise ValueError(
                f"CSV requires x,y columns; found {reader.fieldnames}"
            )

        points = []

        for line_no, row in enumerate(reader, 2):

            try:

                x = float(row[xc])
                y = float(row[yc])

                yaw = (
                    float(row[cols["yaw"]])
                    if "yaw" in cols
                    and row.get(cols["yaw"], "").strip()
                    else None
                )

            except (TypeError, ValueError):

                raise ValueError(
                    f"Invalid numeric value in CSV line {line_no}"
                )

            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError(
                    f"Non-finite x/y in CSV line {line_no}"
                )

            if yaw is not None and not math.isfinite(yaw):
                raise ValueError(
                    f"Non-finite yaw in CSV line {line_no}"
                )

            points.append((x, y, yaw))

    if len(points) >= 3:

        if math.hypot(
            points[-1][0] - points[0][0],
            points[-1][1] - points[0][1]
        ) < 1e-4:

            points.pop()

    if len(points) < 3:
        raise ValueError(
            "At least three distinct path points are required."
        )

    return points


def transform_path_to_odom(path, x0, y0, yaw0):

    """
    Transformacja punktów trajektorii WORLD -> ODOM.

    x0, y0, yaw0:
        Pozycja początku układu ODOM wyrażona w WORLD.

    UWAGA:
        Parametry muszą pochodzić z rzeczywistej transformacji
        pomiędzy WORLD i ODOM, a nie być zgadywane.
    """

    transformed = []

    c = math.cos(yaw0)
    s = math.sin(yaw0)

    for x, y, yaw in path:

        dx = x - x0
        dy = y - y0

        x_odom = c * dx + s * dy
        y_odom = -s * dx + c * dy

        if yaw is not None:

            yaw_odom = normalize_angle(yaw - yaw0)

        else:

            yaw_odom = None

        transformed.append(
            (x_odom, y_odom, yaw_odom)
        )

    return transformed


class RoboRacerFollower(Node):

    def __init__(self):

        super().__init__(
            "roboracer_plus_trajectory_follower"
        )

        # Topics

        self.declare_parameter(
            "csv_path", ""
        )

        self.declare_parameter(
            "odom_topic", "/odom"
        )

        self.declare_parameter(
            "cmd_vel_topic", "/cmd_vel"
        )

        # Controller

        self.declare_parameter(
            "lookahead_distance", 0.65
        )

        self.declare_parameter(
            "target_speed", 0.20
        )

        self.declare_parameter(
            "max_speed", 0.25
        )

        self.declare_parameter(
            "max_yaw_rate", 0.60
        )

        self.declare_parameter(
            "kp_speed", 0.8
        )

        self.declare_parameter(
            "ki_speed", 0.04
        )

        self.declare_parameter(
            "kd_speed", 0.01
        )

        self.declare_parameter(
            "control_rate", 20.0
        )

        # Start protection

        self.declare_parameter(
            "start_tolerance", 0.75
        )

        self.declare_parameter(
            "require_near_start", True
        )

        # WORLD -> ODOM transformation

        self.declare_parameter(
            "world_offset_x", -2.0
        )

        self.declare_parameter(
            "world_offset_y", 3.3
        )

        self.declare_parameter(
            "world_offset_yaw", 1.57
        )

        # Load CSV

        csv_path = str(
            self.get_parameter("csv_path").value
        )

        if not csv_path:

            raise ValueError(
                "Set -p csv_path:=/absolute/path/to/trajektoria_plus.csv"
            )

        self.path = load_path(csv_path)

        # Transform trajectory

        x0 = float(
            self.get_parameter("world_offset_x").value
        )

        y0 = float(
            self.get_parameter("world_offset_y").value
        )

        yaw0 = float(
            self.get_parameter("world_offset_yaw").value
        )

        self.path = transform_path_to_odom(
            self.path,
            x0,
            y0,
            yaw0
        )

        self.get_logger().info(
            f"Applied WORLD -> ODOM transform: "
            f"x={x0:.3f}, y={y0:.3f}, yaw={yaw0:.3f}"
        )

        # Parameters

        self.lookahead = float(
            self.get_parameter("lookahead_distance").value
        )

        self.target_speed = float(
            self.get_parameter("target_speed").value
        )

        self.max_speed = float(
            self.get_parameter("max_speed").value
        )

        self.max_yaw_rate = abs(
            float(self.get_parameter("max_yaw_rate").value)
        )

        self.kp = float(
            self.get_parameter("kp_speed").value
        )

        self.ki = float(
            self.get_parameter("ki_speed").value
        )

        self.kd = float(
            self.get_parameter("kd_speed").value
        )

        self.start_tolerance = float(
            self.get_parameter("start_tolerance").value
        )

        self.require_near_start = bool(
            self.get_parameter("require_near_start").value
        )

        if (
            self.lookahead <= 0
            or self.target_speed <= 0
            or self.max_speed <= 0
        ):

            raise ValueError(
                "lookahead_distance, target_speed and max_speed must be positive"
            )

        # State

        self.pose = None

        self.speed_measured = 0.0

        self.nearest_idx = 0

        self.integral = 0.0

        self.previous_error = 0.0

        self.previous_time = None

        self.armed = False

        self.last_status = self.get_clock().now()

        # ROS subscribers

        self.sub = self.create_subscription(
            Odometry,
            str(self.get_parameter("odom_topic").value),
            self.odom_cb,
            10
        )

        # ROS publisher

        self.pub = self.create_publisher(
            Twist,
            str(self.get_parameter("cmd_vel_topic").value),
            10
        )

        # Timer

        rate = max(
            1.0,
            float(self.get_parameter("control_rate").value)
        )

        self.timer = self.create_timer(
            1.0 / rate,
            self.control
        )

        sx, sy, syaw = self.path[0]

        self.get_logger().info(
            f"Loaded trajectory: {len(self.path)} points. "
            f"Start=({sx:.2f}, {sy:.2f}), "
            f"yaw={syaw if syaw is not None else 'n/a'}."
        )

        self.get_logger().info(
            "Waiting for odometry."
        )

    def odom_cb(self, msg):

        p = msg.pose.pose.position

        self.pose = (
            p.x,
            p.y,
            yaw_from_quaternion(
                msg.pose.pose.orientation
            )
        )

        self.speed_measured = float(
            msg.twist.twist.linear.x
        )

    def publish_stop(self):

        if self.pub is not None:

            self.pub.publish(Twist())

    def choose_target(self, x, y):

        n = len(self.path)

        window = min(n, 100)

        candidates = [
            (self.nearest_idx + k) % n
            for k in range(window)
        ]

        idx = min(
            candidates,
            key=lambda i:
            (self.path[i][0] - x)**2
            + (self.path[i][1] - y)**2
        )

        self.nearest_idx = idx

        target = idx

        for _ in range(n):

            tx, ty, _ = self.path[target]

            if math.hypot(
                tx - x,
                ty - y
            ) >= self.lookahead:

                break

            target = (target + 1) % n

        return target

    def control(self):

        if self.pose is None:
            return

        x, y, yaw = self.pose

        start_x, start_y, _ = self.path[0]

        # Start protection

        if not self.armed:

            start_dist = math.hypot(
                start_x - x,
                start_y - y
            )

            if (
                self.require_near_start
                and start_dist > self.start_tolerance
            ):

                self.publish_stop()

                now = self.get_clock().now()

                if (
                    now - self.last_status
                ).nanoseconds / 1e9 > 3.0:

                    self.get_logger().warning(
                        f"Not starting: car is "
                        f"{start_dist:.2f} m from path start "
                        f"({start_x:.2f}, {start_y:.2f})."
                    )

                    self.last_status = now

                return

            self.armed = True

            self.get_logger().info(
                "Car is near trajectory start. Tracking enabled."
            )

        # Time step

        now = self.get_clock().now()

        dt = (
            (now - self.previous_time).nanoseconds / 1e9
            if self.previous_time
            else 0.05
        )

        self.previous_time = now

        dt = min(max(dt, 0.001), 0.2)

        # Pure Pursuit

        target_idx = self.choose_target(x, y)

        tx, ty, _ = self.path[target_idx]

        dx = tx - x

        dy = ty - y

        local_y = (
            -math.sin(yaw) * dx
            + math.cos(yaw) * dy
        )

        distance_sq = max(
            dx*dx + dy*dy,
            1e-4
        )

        curvature = (
            2.0 * local_y / distance_sq
        )

        # PID speed controller

        v_ref = min(
            self.target_speed,
            self.max_speed
        )

        error = (
            v_ref - self.speed_measured
        )

        self.integral = max(
            -1.5,
            min(
                1.5,
                self.integral + error * dt
            )
        )

        derivative = (
            error - self.previous_error
        ) / dt

        self.previous_error = error

        v_cmd = (
            v_ref
            + self.kp * error
            + self.ki * self.integral
            + self.kd * derivative
        )

        v_cmd = max(
            0.0,
            min(self.max_speed, v_cmd)
        )

        yaw_rate = (
            v_cmd * curvature
        )

        yaw_rate = max(
            -self.max_yaw_rate,
            min(self.max_yaw_rate, yaw_rate)
        )

        # Publish command

        cmd = Twist()

        cmd.linear.x = v_cmd

        cmd.angular.z = yaw_rate

        self.pub.publish(cmd)

        # Status

        if (
            now - self.last_status
        ).nanoseconds / 1e9 > 3.0:

            self.get_logger().info(
                f"path index={self.nearest_idx}/{len(self.path)-1}, "
                f"v={v_cmd:.2f} m/s, "
                f"yaw_rate={yaw_rate:.2f} rad/s"
            )

            self.last_status = now

    def destroy_node(self):

        try:
            self.publish_stop()
        except Exception:
            pass

        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = None

    try:

        node = RoboRacerFollower()

        rclpy.spin(node)

    except (ValueError, FileNotFoundError) as exc:

        print(
            f"[roboracer_follower] Configuration error: {exc}"
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
