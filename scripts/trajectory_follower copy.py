#!/usr/bin/env python3
"""ROS 2 Jazzy follower for the closed-loop RoboRacer PLUS track.

Input CSV: x,y,yaw (yaw is accepted and checked; path tracking uses x/y).
Controller: Pure Pursuit for steering direction + PID for longitudinal speed.
ROS I/O: /odom (nav_msgs/Odometry) -> /cmd_vel (geometry_msgs/Twist).

IMPORTANT: this publishes Twist.angular.z as desired yaw rate [rad/s].
Verify that the actual Gazebo AckermannSteering plugin interprets cmd_vel this way.
"""
import csv
import math
from pathlib import Path

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist


def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w*q.z + q.x*q.y),
                      1.0 - 2.0 * (q.y*q.y + q.z*q.z))


def load_path(filename):
    path = Path(filename).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"CSV trajectory not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV requires a header row.")
        cols = {c.strip().lower(): c for c in reader.fieldnames}
        xc, yc = cols.get("x"), cols.get("y")
        if xc is None or yc is None:
            raise ValueError(f"CSV requires x,y columns; found {reader.fieldnames}")
        points = []
        for line_no, row in enumerate(reader, 2):
            try:
                x, y = float(row[xc]), float(row[yc])
                yaw = float(row[cols["yaw"]]) if "yaw" in cols and row.get(cols["yaw"], "").strip() else None
            except (TypeError, ValueError):
                raise ValueError(f"Invalid numeric value in CSV line {line_no}")
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError(f"Non-finite x/y in CSV line {line_no}")
            if yaw is not None and not math.isfinite(yaw):
                raise ValueError(f"Non-finite yaw in CSV line {line_no}")
            points.append((x, y, yaw))

    # This supplied trajectory is closed: the last point duplicates the first.
    if len(points) >= 3 and math.hypot(points[-1][0]-points[0][0],
                                       points[-1][1]-points[0][1]) < 1e-4:
        points.pop()
    if len(points) < 3:
        raise ValueError("At least three distinct path points are required.")
    return points


class RoboRacerFollower(Node):
    def __init__(self):
        super().__init__("roboracer_plus_trajectory_follower")
        self.declare_parameter("csv_path", "")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("lookahead_distance", 0.65)
        self.declare_parameter("target_speed", 0.25)
        self.declare_parameter("max_speed", 0.35)
        self.declare_parameter("max_yaw_rate", 0.60)
        self.declare_parameter("kp_speed", 0.8)
        self.declare_parameter("ki_speed", 0.04)
        self.declare_parameter("kd_speed", 0.01)
        self.declare_parameter("control_rate", 20.0)
        self.declare_parameter("start_tolerance", 0.75)
        self.declare_parameter("require_near_start", True)

        csv_path = str(self.get_parameter("csv_path").value)
        if not csv_path:
            raise ValueError("Set -p csv_path:=/absolute/path/to/trajektoria_plus.csv")
        self.path = load_path(csv_path)

        self.lookahead = float(self.get_parameter("lookahead_distance").value)
        self.target_speed = float(self.get_parameter("target_speed").value)
        self.max_speed = float(self.get_parameter("max_speed").value)
        self.max_yaw_rate = abs(float(self.get_parameter("max_yaw_rate").value))
        self.kp = float(self.get_parameter("kp_speed").value)
        self.ki = float(self.get_parameter("ki_speed").value)
        self.kd = float(self.get_parameter("kd_speed").value)
        self.start_tolerance = float(self.get_parameter("start_tolerance").value)
        self.require_near_start = bool(self.get_parameter("require_near_start").value)

        if self.lookahead <= 0 or self.target_speed <= 0 or self.max_speed <= 0:
            raise ValueError("lookahead_distance, target_speed and max_speed must be positive")

        self.pose = None
        self.speed_measured = 0.0
        self.nearest_idx = 0
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = None
        self.armed = False
        self.last_status = self.get_clock().now()

        self.sub = self.create_subscription(
            Odometry, str(self.get_parameter("odom_topic").value), self.odom_cb, 10)
        self.pub = self.create_publisher(
            Twist, str(self.get_parameter("cmd_vel_topic").value), 10)
        rate = max(1.0, float(self.get_parameter("control_rate").value))
        self.timer = self.create_timer(1.0/rate, self.control)

        sx, sy, syaw = self.path[0]
        self.get_logger().info(
            f"Loaded closed path: {len(self.path)} points. "
            f"Start=({sx:.2f}, {sy:.2f}), yaw={syaw if syaw is not None else 'n/a'}. "
            "Waiting for odometry."
        )

    def odom_cb(self, msg):
        p = msg.pose.pose.position
        self.pose = (p.x, p.y, yaw_from_quaternion(msg.pose.pose.orientation))
        self.speed_measured = float(msg.twist.twist.linear.x)

    def publish_stop(self):
        self.pub.publish(Twist())

    def choose_target(self, x, y):
        n = len(self.path)
        # Search a forward window with wrap-around, preserving progress around the loop.
        window = min(n, 100)
        candidates = [(self.nearest_idx + k) % n for k in range(window)]
        idx = min(candidates, key=lambda i:
                  (self.path[i][0]-x)**2 + (self.path[i][1]-y)**2)
        self.nearest_idx = idx

        target = idx
        for _ in range(n):
            tx, ty, _ = self.path[target]
            if math.hypot(tx-x, ty-y) >= self.lookahead:
                break
            target = (target + 1) % n
        return target

    def control(self):
        if self.pose is None:
            return

        x, y, yaw = self.pose
        start_x, start_y, _ = self.path[0]

        # Do not unexpectedly drive if the car was spawned somewhere unrelated.
        if not self.armed:
            start_dist = math.hypot(start_x-x, start_y-y)
            if self.require_near_start and start_dist > self.start_tolerance:
                self.publish_stop()
                now = self.get_clock().now()
                if (now-self.last_status).nanoseconds/1e9 > 3.0:
                    self.get_logger().warning(
                        f"Not starting: car is {start_dist:.2f} m from path start "
                        f"({start_x:.2f}, {start_y:.2f}). Move/spawn it near the start."
                    )
                    self.last_status = now
                return
            self.armed = True
            self.get_logger().info("Car is near trajectory start. Tracking enabled.")

        now = self.get_clock().now()
        dt = (now-self.previous_time).nanoseconds/1e9 if self.previous_time else 0.05
        self.previous_time = now
        dt = min(max(dt, 0.001), 0.2)

        target_idx = self.choose_target(x, y)
        tx, ty, _ = self.path[target_idx]

        # Pure Pursuit curvature in the vehicle coordinate frame.
        dx, dy = tx-x, ty-y
        local_y = -math.sin(yaw)*dx + math.cos(yaw)*dy
        distance_sq = max(dx*dx + dy*dy, 1e-4)
        curvature = 2.0*local_y/distance_sq

        v_ref = min(self.target_speed, self.max_speed)
        error = v_ref-self.speed_measured
        self.integral = max(-1.5, min(1.5, self.integral + error*dt))
        derivative = (error-self.previous_error)/dt
        self.previous_error = error

        v_cmd = v_ref + self.kp*error + self.ki*self.integral + self.kd*derivative
        v_cmd = max(0.0, min(self.max_speed, v_cmd))
        yaw_rate = max(-self.max_yaw_rate, min(self.max_yaw_rate, v_cmd*curvature))

        cmd = Twist()
        cmd.linear.x = v_cmd
        cmd.angular.z = yaw_rate
        self.pub.publish(cmd)

        if (now-self.last_status).nanoseconds/1e9 > 3.0:
            self.get_logger().info(
                f"path index={self.nearest_idx}/{len(self.path)-1}, "
                f"v={v_cmd:.2f} m/s, yaw_rate={yaw_rate:.2f} rad/s"
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
        print(f"[roboracer_follower] Configuration error: {exc}")
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.publish_stop()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
