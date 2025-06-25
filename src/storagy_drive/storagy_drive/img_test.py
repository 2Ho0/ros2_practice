import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch
import torch.optim as optim
from torchvision import transforms as T

from tiny_yolov1net_squeezenet import Tiny_YoloV1_SqueezeNet
from yolov1_utils import non_max_suppression, cellboxes_to_boxes
from custom_transform import draw_bounding_box

# Load model and necessary setup outside class
device = torch.device("cpu")
transform = T.Compose([T.ToTensor()])
model = Tiny_YoloV1_SqueezeNet(S=7, B=2, C=20).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=5e-4)

checkpoint = torch.load("cpts/squeezenet_tiny_adj_lr_yolov1.cpt", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model.eval()
torch.set_num_threads(6)

# CvBridge instance
bridge = CvBridge()


class Yolov1ImageNode(Node):

    def __init__(self):
        super().__init__('yolov1_image_node')

        self.img_subscription = self.create_subscription(
            Image,
            '/intel_realsense_r200_depth/image_raw',
            self.listener_callback,
            10
        )
        self.img_publisher = self.create_publisher(Image, '/binarized_img', 10)

    def listener_callback(self, msg):
        frame = bridge.imgmsg_to_cv2(msg, msg.encoding)
        frame_resized = cv2.resize(frame, (448, 448), interpolation=cv2.INTER_AREA)

        input_tensor = transform(frame_resized).unsqueeze(0).to(device)

        with torch.no_grad():
            preds = model(input_tensor)

        boxes = cellboxes_to_boxes(preds)
        bboxes = non_max_suppression(boxes[0], iou_threshold=0.5, threshold=0.4, boxformat="midpoints")
        frame_disp = draw_bounding_box(frame_resized, bboxes, test=True)

        if bboxes:
            largest = max(bboxes, key=lambda b: b[4] * b[5])
            class_id = int(largest[0])
            conf = round(largest[1], 2)
            x, y = round(largest[2], 2), round(largest[3], 2)
            w, h = round(largest[4], 2), round(largest[5], 2)
            area = round(w * h, 2)
            self.get_logger().info(
                f"Detected class={class_id}, conf={conf}, center=({x}, {y}), size=({w}x{h}), area={area}"
            )
        else:
            self.get_logger().info("No object detected.")

        out_msg = bridge.cv2_to_imgmsg(frame_disp, encoding="bgr8")
        self.img_publisher.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = Yolov1ImageNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()