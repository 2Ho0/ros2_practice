import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time

class DetectionProcessor(Node):
    def __init__(self):
        super().__init__('detection_processor')
        self.subscription = self.create_subscription(
            String,
            '/yolo_output/detections',
            self.listener_callback,
            10)
       
        self.buffer = []  # (x, y) tuples
        self.start_time = time.time()

    def listener_callback(self, msg):
        detections = msg.data.strip('[]').replace("'", "").split(', ')
        current_time = time.time()

        for item in detections:
            if ':' in item:
                class_id_str, coords_str = item.split(':')
                if class_id_str == '14':  # 필터: class id 14
                    try:
                        x_str, y_str = coords_str.split(',')
                        x = float(x_str)
                        y = float(y_str)
                        self.buffer.append((x, y))
                    except ValueError:
                        continue  # malformed data, skip

        # 1초 동안 평균 계산
        if current_time - self.start_time > 1.0:
            if self.buffer:
                avg_x = sum(x for x, _ in self.buffer) / len(self.buffer)
                avg_y = sum(y for _, y in self.buffer) / len(self.buffer)
                self.get_logger().info(f"Class 14 평균 좌표: x={avg_x:.3f}, y={avg_y:.3f}")
            else:
                self.get_logger().info("Class 14 객체 없음")
            self.buffer.clear()
            self.start_time = current_time


def main(args=None):
    rclpy.init(args=args)
    processor = DetectionProcessor()
    rclpy.spin(processor)
    processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()