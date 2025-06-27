import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Image
from cv_bridge import CvBridge
import math
import time
import numpy as np
import cv2

class DetectionProcessor(Node):
    def __init__(self):
        super().__init__('detection_processor')

        self.bridge = CvBridge()
        self.latest_depth = None
        self.latest_distance = None
        self.avoid_angular = 0.0
        self.prev_valid_depth = None
        self.emergency_stop = False

        self.Kp_x = 1.5
        self.Ki_x = 0.0
        self.Kd_x = 0.008
        self.Kp_speed = 0.5
        self.Ki_speed = 0.0
        self.Kd_speed = 0.0

        self.dt = 1 / 30.0
        self.integral = 0.0
        self.prev_error_x = 0.0
        self.prev_error_distance = 0.0

        self.twist = Twist()

        self.buffer = []
        self.start_time = time.time()
        self.last_seen_time = None
        self.missing_threshold = 0.3

        self.downsample_gap = 10
        self.max_sight = 1.0
        self.max_gap_safe_dist = 0.5

        self.create_subscription(String, '/yolo_output/detections', self.listener_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.laserscan_callback, 10)
        self.create_subscription(Image, '/camera/depth/image_raw', self.depth_callback, 10)

        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

    def PID_controller(self, error, prev_error, Kp, Ki, Kd, dt):
        P = Kp * error
        self.integral += Ki * error * dt
        D = Kd * (error - prev_error) / dt
        return P + self.integral + D

    def preprocess_lidar(self, ranges):
        n = len(ranges) // self.downsample_gap
        proc_ranges = np.zeros(n)
        for i in range(n):
            chunk = ranges[i * self.downsample_gap : (i + 1) * self.downsample_gap]
            valid = [r for r in chunk if not math.isinf(r) and not math.isnan(r)]
            avg = sum(valid) / len(valid) if valid else self.max_sight
            proc_ranges[i] = min(avg, self.max_sight)
        return proc_ranges

    def find_max_gap(self, free_space_ranges):
        longest_streak = 0
        streak = 0
        end_index = 0
        for i, r in enumerate(free_space_ranges):
            if r > self.max_gap_safe_dist:
                streak += 1
                if streak > longest_streak:
                    longest_streak = streak
                    end_index = i + 1
            else:
                streak = 0
        start_index = end_index - longest_streak
        return start_index, end_index

    def find_best_point(self, start_i, end_i):
        return (start_i + end_i) / 2.0

    def laserscan_callback(self, msg: LaserScan):
        angle_range = math.radians(60)
        center_index = int((0.0 - msg.angle_min) / msg.angle_increment)
        half_range = int(angle_range / msg.angle_increment)
        start_idx = max(0, center_index - half_range)
        end_idx = min(len(msg.ranges), center_index + half_range)

        sub_ranges = msg.ranges[start_idx:end_idx]
        start_angle = msg.angle_min + start_idx * msg.angle_increment

        proc_ranges = self.preprocess_lidar(sub_ranges)
        start_max_gap, end_max_gap = self.find_max_gap(proc_ranges)
        best_i = self.find_best_point(start_max_gap, end_max_gap)

        angle_per_index = self.downsample_gap * msg.angle_increment
        best_angle = start_angle + best_i * angle_per_index
        self.avoid_angular = best_angle

        valid = [r for r in sub_ranges if not math.isinf(r) and not math.isnan(r)]
        self.latest_distance = sum(valid) / len(valid) if valid else None

    def depth_callback(self, msg):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            if self.latest_depth.dtype != np.uint16:
                return

            h, w = self.latest_depth.shape
            cy = h // 2
            roi = self.latest_depth[max(cy - 10, 0):min(cy + 10, h), :]
            valid = roi[(roi >= 200) & (roi <= 3000)]

            if valid.size > 0:
                valid_m = valid.astype(np.float32) / 1000.0
                close_ratio = np.sum(valid_m <= 0.3) / valid_m.size
                if close_ratio >= 0.1:
                    self.emergency_stop = True
                else:
                    self.emergency_stop = False
        except Exception as e:
            self.get_logger().warn(f"[Depth 변환 오류] {e}")

    def listener_callback(self, msg):
        detections = msg.data.strip().split(';')
        current_time = time.time()

        for item in detections:
            parts = item.strip().split(':')
            if len(parts) != 2:
                continue
            class_id_str, coords_str = parts

            if class_id_str.strip() == '14':
                try:
                    coords = list(map(float, coords_str.strip().split(',')))
                    if len(coords) != 4:
                        continue
                    cx, cy, wx, wy = coords
                    self.buffer.append((cx, cy, wx, wy))
                except ValueError:
                    continue

        if current_time - self.start_time > 0.1:
            self.start_time = current_time

            if self.emergency_stop:
                self.twist.linear.x = 0.0
                self.twist.angular.z = 0.0
                self.cmd_vel_pub.publish(self.twist)
                self.get_logger().warn("[긴급정지] Depth ROI 조건 만족")
                return

            if self.buffer:
                self.last_seen_time = current_time
                avg_cx = sum(x for x, _, _, _ in self.buffer) / len(self.buffer)
                avg_cy = sum(y for _, y, _, _ in self.buffer) / len(self.buffer)
                avg_wx = sum(w for _, _, w, _ in self.buffer) / len(self.buffer)
                self.buffer.clear()

                img_w, img_h = 448, 448
                cx_px = int(avg_cx * img_w)
                cy_px = int(avg_cy * img_h)
                wx_px = int(avg_wx * img_w)

                error_x = avg_cx - 0.5
                depth = None

                if self.latest_depth is not None:
                    depth_resized = cv2.resize(self.latest_depth, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                    roi_half = int(max(3, min(15, wx_px / 2)))
                    x1 = max(0, cx_px - roi_half)
                    x2 = min(img_w - 1, cx_px + roi_half)
                    y1 = max(0, cy_px - roi_half)
                    y2 = min(img_h - 1, cy_px + roi_half)
                    if x2 > x1 and y2 > y1:
                        roi = depth_resized[y1:y2, x1:x2]
                        valid = roi[(roi >= 200) & (roi <= 3000)]
                        if valid.size > 0:
                            depth = float(np.mean(valid)) / 1000.0

                # Depth 보정: 갑자기 0이 되면 이전 값 유지
                if depth is not None and depth > 0.0:
                    self.prev_valid_depth = depth
                elif self.prev_valid_depth is not None and (depth is None or depth == 0.0):
                    depth = self.prev_valid_depth

                if depth is not None and depth < 3.0:
                    self.get_logger().info(f"[class 14] Depth: {depth:.2f} m")
                    if depth <= 0.5:
                        self.twist.linear.x = 0.0
                        self.twist.angular.z = 0.0
                        self.get_logger().warn("Depth 너무 가까움: 정지")
                    else:
                        error_distance = depth - 0.5
                        speed = self.PID_controller(error_distance, self.prev_error_distance,
                                                    self.Kp_speed, self.Ki_speed, self.Kd_speed, self.dt)
                        self.prev_error_distance = error_distance
                        self.twist.linear.x = min(speed, 0.3)
                        self.twist.angular.z = -self.PID_controller(error_x, self.prev_error_x,
                                                                    self.Kp_x, self.Ki_x, self.Kd_x, self.dt)
                        self.prev_error_x = error_x
                elif self.latest_distance is not None and self.latest_distance > 0.5:
                    error_distance = self.latest_distance - 0.5
                    speed = self.PID_controller(error_distance, self.prev_error_distance,
                                                self.Kp_speed, self.Ki_speed, self.Kd_speed, self.dt)
                    self.prev_error_distance = error_distance
                    self.twist.linear.x = min(speed, 0.3)
                    if abs(error_x) > 0.05:
                        angular = self.PID_controller(error_x, self.prev_error_x,
                                                      self.Kp_x, self.Ki_x, self.Kd_x, self.dt)
                        self.prev_error_x = error_x
                        self.twist.angular.z = -angular
                    else:
                        self.twist.angular.z = 0.0
                else:
                    self.twist.linear.x = 0.0
                    self.twist.angular.z = 0.0
                    self.get_logger().info("너무 가까움 또는 정보 부족")
            else:
                if self.last_seen_time is None or (current_time - self.last_seen_time) > self.missing_threshold:
                    if self.latest_distance is not None and self.latest_distance > 0.5:
                        error_distance = self.latest_distance - 0.5
                        speed = self.PID_controller(error_distance, self.prev_error_distance,
                                                    self.Kp_speed, self.Ki_speed, self.Kd_speed, self.dt)
                        self.prev_error_distance = error_distance
                        self.twist.linear.x = min(speed, 0.3)
                    else:
                        self.twist.linear.x = 0.0
                    self.twist.angular.z = self.avoid_angular
                    self.get_logger().info("YOLO 미탐지 상태: LIDAR 회피")
                else:
                    self.get_logger().info("YOLO 놓침: 유예 시간")

            self.cmd_vel_pub.publish(self.twist)

def main(args=None):
    rclpy.init(args=args)
    node = DetectionProcessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()