import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32MultiArray
from cv_bridge import CvBridge
import numpy as np

class YOLODepthDistance(Node):
    def __init__(self):
        super().__init__('yolo_depth_distance_node')
        self.bridge = CvBridge()
        self.latest_depth = None

        self.sub_depth = self.create_subscription(
            Image, '/camera/depth/image_raw', self.depth_callback, 10)

        self.sub_yolo = self.create_subscription(
            String, '/yolo_output/detections', self.yolo_callback, 10)

        self.pub = self.create_publisher(
            Float32MultiArray, '/yolo_depth/avg_box_distances', 10)

    def depth_callback(self, msg):
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def yolo_callback(self, msg):
        if self.latest_depth is None or not msg.data.strip():
            return

        depth = self.latest_depth
        h, w = depth.shape

        results = []
        detections = msg.data.strip().split(';')

        for det in detections:
            try:
                cls, val = det.strip().split(':')
                cx, cy, wx, wy = map(float, val.split(','))
                cx, cy = int(cx), int(cy)
                wx, wy = int(wx), int(wy)

                x1 = max(int(cx - wx / 2), 0)
                x2 = min(int(cx + wx / 2), w - 1)
                y1 = max(int(cy - wy / 2), 0)
                y2 = min(int(cy + wy / 2), h - 1)

                roi = depth[y1:y2, x1:x2]
                valid = roi[(roi > 0.2) & (roi <= 1.0)]

                avg = float(np.mean(valid)) if valid.size > 0 else 1.0
                avg = min(avg, 1.0)  # 혹시라도 넘는 경우
                results.append(avg)

                self.get_logger().info(f"[class {cls}] 거리: {avg:.3f} m")

            except Exception as e:
                self.get_logger().warn(f"파싱 오류: {det} → {e}")
                results.append(1.0)  # 오류 시에도 1.0으로 채움

        msg_out = Float32MultiArray(data=results)
        self.pub.publish(msg_out)

def main(args=None):
    rclpy.init(args=args)
    node = YOLODepthDistance()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
