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

        self.Kp_x = 1.5
        self.Ki_x = 0.0
        self.Kd_x = 0.008

        self.Kp_speed = 0.002
        self.Ki_speed = 0.0
        self.Kd_speed = 0.0

        self.dt = 1 / 30
        self.ws = 0.0
        self.speed_min = 0.0
        self.integral = 0.0
        self.prev_error_x = 0.0
        self.prev_error_distance = 0.0

        self.twist = Twist()
        self.latest_distance = None

        self.subscription = self.create_subscription(
            String,
            '/yolo_output/detections',
            self.listener_callback,
            10)

        self.create_subscription(
            LaserScan, '/scan', self.laserscan_callback, 10)

        self.create_subscription(
            Image, '/camera/depth/image_raw', self.depth_callback, 10)

        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        self.buffer = []
        self.start_time = time.time()

    def depth_callback(self, msg):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().warn(f"[Depth] 변환 실패: {e}")

    def laserscan_callback(self, msg: LaserScan):
        center_index = int((0.0 - msg.angle_min) / msg.angle_increment)
        angle_range = int(math.radians(10) / msg.angle_increment)

        start_idx = max(0, center_index - angle_range)
        end_idx = min(len(msg.ranges), center_index + angle_range)

        distances = [r for r in msg.ranges[start_idx:end_idx] if not math.isinf(r) and not math.isnan(r)]
        self.latest_distance = sum(distances) / len(distances) if distances else None

    def PID_controller(self, error, prev_error, Kp, Ki, Kd, dt=1/30):
        P = Kp * error
        self.integral += Ki * error * dt
        D = Kd * (error - prev_error) / dt
        return P + self.integral + D

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

            if self.buffer:
                avg_cx = sum(x for x, _, _, _ in self.buffer) / len(self.buffer)
                avg_cy = sum(y for _, y, _, _ in self.buffer) / len(self.buffer)
                avg_wx = sum(w for _, _, w, _ in self.buffer) / len(self.buffer)
                avg_wy = sum(h for _, _, _, h in self.buffer) / len(self.buffer)
                self.buffer.clear()

                # 정규화된 좌표 → 픽셀 단위 변환
                img_w, img_h = 448, 448
                cx_px = int(avg_cx * img_w)
                cy_px = int(avg_cy * img_h)
                wx_px = int(avg_wx * img_w)

                error_x = (avg_cx - 0.5)

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
                            depth = float(np.mean(valid)) / 1000.0  # mm → m

                if depth is not None and depth < 3.0:
                    self.get_logger().info(f"[class 14] 평균 거리 (depth): {depth:.2f} m")
                    if depth > 0.4:
                        speed = self.PID_controller(depth - 0.2, self.prev_error_distance,
                                                    self.Kp_speed, self.Ki_speed, self.Kd_speed)
                        self.prev_error_distance = depth - 0.2
                        self.twist.linear.x = min(speed, 0.2)
                        self.twist.angular.z = -self.PID_controller(error_x, self.prev_error_x,
                                                                    self.Kp_x, self.Ki_x, self.Kd_x)
                        self.prev_error_x = error_x
                        self.get_logger().info("Depth 기반 감속 중")
                    else:
                        self.twist.linear.x = 0.0
                        self.twist.angular.z = 0.0
                        self.get_logger().info("Depth 한계 도달. 정지")
                elif self.latest_distance is not None and self.latest_distance > 0.5:
                    self.twist.linear.x = 0.3 / (avg_wx if avg_wx > 0 else 1.0)
                    if abs(error_x) > 0.05:
                        angular = self.PID_controller(error_x, self.prev_error_x,
                                                      self.Kp_x, self.Ki_x, self.Kd_x)
                        self.prev_error_x = error_x
                        self.twist.angular.z = -angular
                    self.get_logger().info("Laser 기반 추적 중")
                else:
                    self.twist.linear.x = 0.0
                    self.twist.angular.z = 0.0
                    self.get_logger().info("너무 가까움 또는 정보 부족")

                self.cmd_vel_pub.publish(self.twist)
            else:
                self.get_logger().info("Class 14 객체 없음")

def main(args=None):
    rclpy.init(args=args)
    node = DetectionProcessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()