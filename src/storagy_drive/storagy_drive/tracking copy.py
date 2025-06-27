import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import math
import numpy as np

class DetectionProcessor(Node):
    def __init__(self):
        super().__init__('detection_processor')

        self.Kp_x = 1.5
        self.Ki_x = 0.0
        self.Kd_x = 0.008
        self.Kp_speed = 0.003
        self.Ki_speed = 0.0
        self.Kd_speed = 0.0
        self.dt = 1/30
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
        self.create_subscription(LaserScan, '/scan', self.laserscan_callback, 10)

        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel',10)
        self.buffer = []  # (x, y) tuples
        self.start_time = time.time()

    def PID_controller(self, error, prev_error, Kp, Ki, Kd, dt = 1/30):

            P = Kp * error
            I = self.integral + Ki * error * dt
            D = Kd * (error - prev_error) / dt
            output = P + I + D

            return output


    def laserscan_callback(self, msg: LaserScan):
        center_index = int((0.0- msg.angle_min)/msg.angle_increment)
        angle_range = int(math.radians(10)/msg.angle_increment)

        start_idx = max(0, center_index - angle_range)
        end_idx = min(len(msg.ranges), center_index + angle_range)

        distances = [r for r in msg.ranges[start_idx:end_idx] if not math.isinf(r) and not math.isnan(r)]
        if distances:
            self.latest_distance = sum(distances) / len(distances)
        else:
            self.latest_distance = None

    def listener_callback(self, msg):
        detections = msg.data.strip().split(';')
        current_time = time.time()

        for item in detections:
            parts = item.strip().split(':')
            if len(parts)!=2:
                continue
            class_id_str = parts[0].strip()
            coords_str = parts[1].strip()
            
            if class_id_str == '14':  # 필터: class id 14
                try:
                    coords = coords_str.split(',')
                    if len(coords) != 3:
                        continue
                    x = float(coords[0].strip())
                    y = float(coords[1].strip())
                    self.ws = float(coords[2].strip())
                    self.buffer.append((x, y, self.ws))
                except ValueError:
                    continue  # malformed data, skip

        # 0.05초 동안 평균 계산
        if current_time - self.start_time > 0.1:
            self.start_time = current_time
            if self.buffer:
                avg_x = sum(x for x, _, _ in self.buffer) / len(self.buffer)
                avg_y = sum(y for _, y, _ in self.buffer) / len(self.buffer)
                avg_ws = sum(self.ws for _, _, self.ws in self.buffer) / len(self.buffer)
                self.get_logger().info(f"Class 14 평균 좌표: x={avg_x:.3f}, y={avg_y:.3f}")
                self.get_logger().info(f"Bounding Box: width ={self.ws:.3f}")
                self.buffer.clear()

                error_x = avg_x - 0.5

                if self.latest_distance is not None and self.latest_distance > 0.5:
                    self.speed_min = 0.3
                    if avg_ws == 0:
                        self.twist.linear.x = self.speed_min
                    else:
                        self.twist.linear.x = self.speed_min/avg_ws

                    if  abs(error_x) > 0.05:
                        angular = self.PID_controller(error_x, self.prev_error_x, self.Kp_x, self.Ki_x, self.Kd_x)
                        self.prev_error_x = error_x
                        self.twist.angular.z = -angular

                    self.get_logger().info("따라가는 중")
                else:
                    error_distance = self.latest_distance - 0.5
                    speed = self.PID_controller(error_distance, self.prev_error_distance, self.Kp_speed, self.Ki_speed, self.Kd_speed)
                    self.prev_error_distance = error_distance
                    self.twist.linear.x = speed
                    # self.twist.linear.x = 0.0
                    self.twist.angular.z = 0.0
                    self.get_logger().info("너무 가까워요")
                    
                self.cmd_vel_pub.publish(self.twist)
            
            
            else:
                self.get_logger().info("Class 14 객체 없음")
            


def main(args=None):
    rclpy.init(args=args)
    processor = DetectionProcessor()
    rclpy.spin(processor)
    processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()