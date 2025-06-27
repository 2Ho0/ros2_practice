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

        # PID parameters
        self.Kp_x = 1.5
        self.Ki_x = 0.0
        self.Kd_x = 0.008

        self.Kp_speed = 0.003
        self.Ki_speed = 0.0
        self.Kd_speed = 0.0

        self.Kp_speed_no = 0.0005
        self.Ki_speed_no = 0.0
        self.Kd_speed_no = 0.006


        self.Kp_angular = 1.5
        self.Ki_angular = 0.0
        self.Kd_angular = 0.0
        
        self.dt = 1 / 30.0

        # State variables
        self.ws = 0.0
        self.wy = 0.0
        self.speed = 0.0
        self.angular = 0.0
        self.speed_min = 0.0
        self.prev_angular = 0.0
        self.prev_angular_gap = 0.0
        self.prev_speed = 0.0
        self.prev_speed_gap = 0.0
        
        self.integral = 0.0
        self.prev_error_x = 0.0
        self.prev_error_distance = 0.0

        self.twist = Twist()
        self.latest_distance = None
        self.prev_latest_distance = 0.0
        self.avoid_angular = 0.0
        self.buffer = []
        self.start_time = time.time()

        # LIDAR parameters
        self.downsample_gap = 10
        self.max_sight = 4.0
        self.max_gap_safe_dist = 0.45

        # ROS2 Subscribers & Publishers
        self.sub_yolo = self.create_subscription(
            String,
            '/yolo_output/detections',
            self.listener_callback,
            10
        )
        self.sub_scan = self.create_subscription(
            LaserScan,
            '/scan',
            self.laserscan_callback,
            10
        )
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

    def PID_controller(self, error, prev_error, Kp, Ki, Kd, dt=1 / 30.0):
        P = Kp * error
        I = self.integral + Ki * error * dt
        D = Kd * (error - prev_error) / dt
        return P + I + D

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

    def find_best_point(self, start_i, end_i, ranges):
        return (start_i + end_i) / 2.0

    def laserscan_callback(self, msg: LaserScan):
        # +-10도 범위 설정
        angle_range_human = math.radians(20)
        angle_range_ob = math.radians(70)
        center_index = int((0.0 - msg.angle_min) / msg.angle_increment)
        half_range_human = int(angle_range_human / msg.angle_increment)
        half_range_ob = int(angle_range_ob / msg.angle_increment)
        start_idx_human = max(0, center_index - half_range_human)
        start_idx_ob = max(0, center_index - half_range_ob)
        end_idx_human = min(len(msg.ranges), center_index + half_range_human)
        end_idx_ob = min(len(msg.ranges), center_index + half_range_ob)

        sub_ranges_human = msg.ranges[start_idx_human:end_idx_human]
        sub_ranges_ob = msg.ranges[start_idx_ob:end_idx_ob]
        
        start_angle = msg.angle_min + start_idx_ob * msg.angle_increment

        proc_ranges = self.preprocess_lidar(sub_ranges_ob)
        start_max_gap, end_max_gap = self.find_max_gap(proc_ranges)
        best_i = self.find_best_point(start_max_gap, end_max_gap, proc_ranges)

        angle_per_index = self.downsample_gap * msg.angle_increment
        best_angle = start_angle + best_i * angle_per_index
        self.avoid_angular = best_angle

        valid_distances = [r for r in sub_ranges_human if not math.isinf(r) and not math.isnan(r)]
        self.latest_distance = sum(valid_distances) / len(valid_distances) if valid_distances else None
        self.prev_latest_distance = self.latest_distance

    def listener_callback(self, msg: String):
        detections = msg.data.strip().split(';')
        current_time = time.time()

        for item in detections:
            parts = item.strip().split(':')
            if len(parts) != 2:
                continue
            class_id_str = parts[0].strip()
            coords_str = parts[1].strip()

            if class_id_str == '14':
                try:
                    coords = coords_str.split(',')
                    if len(coords) != 4:
                        continue
                    x = float(coords[0].strip())
                    y = float(coords[1].strip())
                    self.ws = float(coords[2].strip())
                    self.buffer.append((x, y, self.ws, self.wy))
                except ValueError:
                    continue

        if current_time - self.start_time > 0.1:
            self.start_time = current_time

            if self.buffer:
                avg_x = sum(x for x, _, _, _ in self.buffer) / len(self.buffer)
                avg_y = sum(y for _, y, _, _ in self.buffer) / len(self.buffer)
                avg_ws = sum(ws for _, _, ws, _ in self.buffer) / len(self.buffer)
                self.buffer.clear()

                self.get_logger().info(f"[YOLO] 평균 좌표: x={avg_x:.3f}, y={avg_y:.3f}, width={avg_ws:.3f}")
                error_x = avg_x - 0.5

                if self.prev_latest_distance is not None and self.prev_latest_distance > 0.5:
                    self.speed_min = 0.3
                    self.speed = self.speed_min if avg_ws == 0.0 else self.speed_min / avg_ws
                    self.prev_speed = self.speed
                    self.twist.linear.x = self.prev_speed


                    if abs(error_x) > 0.05:
                        self.angular = self.PID_controller(error_x, self.prev_error_x, self.Kp_x, self.Ki_x, self.Kd_x)
                        self.prev_error_x = error_x
                        self.prev_angular = -self.angular
                        self.twist.angular.z = -self.angular
                    else:
                        self.twist.angular.z = 0.0

                    self.get_logger().info("사람 따라갈게요.")
                else:
                    if self.prev_latest_distance is not None:
                        error_distance = self.prev_latest_distance - 0.5
                        speed = self.PID_controller(error_distance, self.prev_error_distance, self.Kp_speed, self.Ki_speed, self.Kd_speed)
                        self.prev_error_distance = error_distance
                        self.speed = speed
                        self.prev_speed = self.speed
                        self.twist.linear.x = self.prev_speed
                        self.twist.angular.z = 0.0
                        self.get_logger().info("너무 가까워요.")
                    else:
                        self.get_logger().info("거리 데이터 없어요. 동작 초기 상태입니다.")
            else:
                if self.prev_latest_distance is not None:
                    speed_gap = self.prev_speed - self.speed_min
                    speed = self.PID_controller(speed_gap, self.prev_speed_gap, self.Kp_speed_no, self.Ki_speed_no, self.Kd_speed_no)
                    self.prev_speed_gap = speed_gap
                    self.twist.linear.x = speed
                else:
                    self.twist.linear.x = self.prev_speed

                angular_gap = self.angular - self.prev_angular
                angular = self.PID_controller(angular_gap, self.prev_angular_gap, self.Kp_angular, self.Ki_angular, self.Kd_angular)
                self.prev_angular_gap = angular_gap

                self.twist.angular.z = angular
                self.get_logger().info("사람 없어요. 자율 회피할게요")

            self.cmd_vel_pub.publish(self.twist)

def main(args=None):
    rclpy.init(args=args)
    processor = DetectionProcessor()
    rclpy.spin(processor)
    processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
