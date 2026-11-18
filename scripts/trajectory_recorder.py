#!/usr/bin/env python3
"""
ROS 2 trajectory recorder for RoboRacer.

Subscribes:
  /odom     nav_msgs/msg/Odometry
  /cmd_vel  geometry_msgs/msg/Twist

Behaviour:
  - Waits for the first /cmd_vel message before starting the recording.
  - Records odometry (x, y, z, yaw) and velocity to CSV.
  - Stops and saves automatically when /cmd_vel messages stop arriving
    for `stop_after` seconds. This allows the recorder to run in its own
    terminal and finish after the controller node is stopped.

The node never publishes commands and does not modify the controller.
"""

import csv
import math
import os
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist


def quaternion_to_yaw(q):
    """Convert an odometry quaternion to yaw in radians."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class TrajectoryRecorder(Node):
    def __init__(self):
        super().__init__('trajectory_recorder')

        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('output_csv', '~/ros2_ws/src/roboracer_sim/config/trajectory_actual.csv')
        self.declare_parameter('stop_after', 3.0)
        self.declare_parameter('min_record_interval', 0.02)

        self.odom_topic = self.get_parameter('odom_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.output_csv = os.path.expanduser(
            self.get_parameter('output_csv').value
        )
        self.stop_after = float(self.get_parameter('stop_after').value)
        self.min_record_interval = float(
            self.get_parameter('min_record_interval').value
        )

        self.started = False
        self.finished = False
        self.last_cmd_time = None
        self.last_record_time = None
        self.row_count = 0

        self.output_path = Path(self.output_csv)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.file = None
        self.writer = None

        self.odom_sub = self.create_subscription(
            Odometry, self.odom_topic, self.odom_callback, 20
        )
        self.cmd_sub = self.create_subscription(
            Twist, self.cmd_vel_topic, self.cmd_callback, 20
        )

        self.watchdog = self.create_timer(0.2, self.check_controller)

        self.get_logger().info('Trajectory recorder is ready.')
        self.get_logger().info(f'Odometry topic: {self.odom_topic}')
        self.get_logger().info(f'Control topic:  {self.cmd_vel_topic}')
        self.get_logger().info(f'CSV destination: {self.output_path}')
        self.get_logger().info(
            'Waiting for /cmd_vel. Start the driving controller in another terminal.'
        )

    def start_recording(self):
        if self.started:
            return

        self.file = open(self.output_path, 'w', newline='', encoding='utf-8')
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            'sample',
            'stamp_sec',
            'stamp_nanosec',
            'time_sec',
            'x_m',
            'y_m',
            'z_m',
            'yaw_rad',
            'yaw_deg',
            'linear_x_mps',
            'linear_y_mps',
            'linear_z_mps',
            'angular_z_radps',
        ])
        self.file.flush()
        self.started = True
        self.record_start_monotonic = time.monotonic()
        self.get_logger().info('Recording started.')

    def cmd_callback(self, _msg):
        # Receipt time is used rather than command magnitude: zero-velocity
        # commands are still valid evidence that the controller is running.
        self.last_cmd_time = time.monotonic()
        if not self.started:
            self.start_recording()

    def odom_callback(self, msg):
        if not self.started or self.finished:
            return

        now = time.monotonic()
        if (self.last_record_time is not None and
                now - self.last_record_time < self.min_record_interval):
            return

        stamp_sec = int(msg.header.stamp.sec)
        stamp_nanosec = int(msg.header.stamp.nanosec)
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        yaw = quaternion_to_yaw(orientation)
        twist = msg.twist.twist

        elapsed = now - self.record_start_monotonic

        self.writer.writerow([
            self.row_count,
            stamp_sec,
            stamp_nanosec,
            f'{elapsed:.6f}',
            f'{position.x:.6f}',
            f'{position.y:.6f}',
            f'{position.z:.6f}',
            f'{yaw:.8f}',
            f'{math.degrees(yaw):.5f}',
            f'{twist.linear.x:.6f}',
            f'{twist.linear.y:.6f}',
            f'{twist.linear.z:.6f}',
            f'{twist.angular.z:.6f}',
        ])

        self.row_count += 1
        self.last_record_time = now

        if self.row_count % 100 == 0:
            self.file.flush()

    def check_controller(self):
        if self.finished or not self.started or self.last_cmd_time is None:
            return

        now = time.monotonic()
        silence = now - self.last_cmd_time
        if silence >= self.stop_after:
            self.finish_recording(
                f'No /cmd_vel messages for {silence:.1f} s; controller appears stopped.'
            )

    def finish_recording(self, reason):
        if self.finished:
            return

        self.finished = True
        if self.file is not None:
            self.file.flush()
            self.file.close()
            self.file = None

        if self.started:
            self.get_logger().info(reason)
            self.get_logger().info(
                f'Recording saved: {self.output_path} ({self.row_count} odometry samples).'
            )
        else:
            self.get_logger().warning(
                'No /cmd_vel message was received. No trajectory CSV was created.'
            )

        self.get_logger().info('Recorder is shutting down.')
        rclpy.shutdown()

    def destroy_node(self):
        if self.file is not None:
            self.file.flush()
            self.file.close()
            self.file = None
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryRecorder()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Recorder stopped manually (Ctrl+C).')
        if node.started and not node.finished:
            node.finish_recording('Recording stopped manually.')
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        node.destroy_node()


if __name__ == '__main__':
    main()
