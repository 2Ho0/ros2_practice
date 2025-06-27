import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch
from torchvision import transforms as T

from models.tiny_yolov1net_squeezenet import Tiny_YoloV1_SqueezeNet
from utils.yolov1_utils import non_max_suppression, cellboxes_to_boxes
from utils.custom_transform import draw_bounding_box

# Setup: device, transform, and model
device = torch.device("cpu")
transform = T.Compose([T.ToTensor()])
torch.set_num_threads(6)

# Load model
model = Tiny_YoloV1_SqueezeNet(S=7, B=2, C=20).to(device)
checkpoint = torch.load("example2/cpts/squeezenet_tiny_adj_lr_yolov1.cpt", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

bridge = CvBridge()

class YoloNode(Node):
    def __init__(self):
        super().__init__('yolov1_node')

        # Subscribe to image input (RGB stream from depth camera)
        self.sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            10
        )

        # Publish YOLO-processed image for visualization
        self.pub = self.create_publisher(
            Image,
            '/yolo_output/image',
            10
        )

    def image_callback(self, msg):
        frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        frame_resized = cv2.resize(frame, (448, 448), interpolation=cv2.INTER_AREA)
        input_tensor = transform(frame_resized).unsqueeze(0).to(device)

        with torch.no_grad():
            preds = model(input_tensor)

        boxes = cellboxes_to_boxes(preds)
        bboxes = non_max_suppression(boxes[0], iou_threshold=0.5, threshold=0.4, boxformat="midpoints")
        result_img = draw_bounding_box(frame_resized, bboxes, test=True)

        if bboxes:
            self.get_logger().info(f"Detected {len(bboxes)} object(s).")
        else:
            self.get_logger().info("No objects detected.")

        ros_img = bridge.cv2_to_imgmsg(result_img, encoding="bgr8")
        self.pub.publish(ros_img)

def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
