import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import torch
from torchvision import transforms as T

from models.tiny_yolov1net_mobilenetv3_small import Tiny_YoloV1_MobileNetV3_Small
from utils.yolov1_utils import non_max_suppression, cellboxes_to_boxes
from utils.custom_transform import draw_bounding_box

# ------------------ Setup ------------------
device = torch.device("cpu")
transform = T.Compose([T.ToTensor()])
torch.set_num_threads(6)

model = Tiny_YoloV1_MobileNetV3_Small(S=7, B=2, C=20).to(device)
checkpoint = torch.load("cpts/mobilenetv3_small_tiny_adj_lr_yolov1.cpt", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

bridge = CvBridge()

# ------------------ ROS 2 Node ------------------
class YoloNode(Node):
    def __init__(self):
        super().__init__('yolov1_node')

        self.sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',  # 입력 이미지 토픽
            self.image_callback,
            10
        )

        self.pub_img = self.create_publisher(
            Image,
            '/yolo_output/image',  # 바운딩 박스 포함된 이미지 출력
            10
        )

        self.pub_txt = self.create_publisher(
            String,
            '/yolo_output/detections',  # class_id와 중심좌표 텍스트 출력
            10
        )

    def image_callback(self, msg):
        frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        frame_resized = cv2.resize(frame, (448, 448), interpolation=cv2.INTER_AREA)
        input_tensor = transform(frame_resized).unsqueeze(0).to(device)

        with torch.no_grad():
            preds = model(input_tensor)

        boxes = cellboxes_to_boxes(preds)
        bboxes = non_max_suppression(boxes[0], iou_threshold=0.5, threshold=0.4, boxformat="midpoints")
        result_img = draw_bounding_box(frame_resized.copy(), bboxes, test=True)

        if bboxes:
            info_list = []
            for box in bboxes:
                class_id = int(box[0])
                cx = round(box[2], 2)
                cy = round(box[3], 2)
                wx = round(box[4], 2)
                wy = round(box[5], 2)
                info_list.append(f"{class_id}:{cx},{cy},{wx},{wy}")
            detection_str = "; ".join(info_list)
            self.pub_txt.publish(String(data=detection_str))
            self.get_logger().info(f"Detected {len(bboxes)} object(s): {detection_str}")
        else:
            self.pub_txt.publish(String(data=""))
            self.get_logger().info("No objects detected.")

        out_msg = bridge.cv2_to_imgmsg(result_img, encoding="bgr8")
        self.pub_img.publish(out_msg)

# ------------------ Main ------------------
def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()