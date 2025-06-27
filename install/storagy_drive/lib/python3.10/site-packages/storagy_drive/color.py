import rclpy
import cv2
import numpy as np


from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image, CompressedImage
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from std_msgs.msg import Float32  # Int32 -> Float32로 변경


class Storagy_Drive(Node):
    def __init__(self):
        super().__init__('storagy_drive')

        self.declare_parameter("image_topic_name", "/camera/color/image_raw/compressed")
        self.declare_parameter("visualization", True)

        self.image_topic_name = self.get_parameter("image_topic_name").get_parameter_value().string_value
        self.visualization = self.get_parameter("visualization").get_parameter_value().bool_value

        # 차선 검출 색상 임계값 설정
        self.YELLOW_LOW_TH = np.array([0, 90, 100])
        self.YELLOW_HIGH_TH = np.array([60, 220, 255])

        # 동적 파라미터 설정
        self.add_on_set_parameters_callback(self.reconfigure_callback)

        # OpenCV Bridge
        self.cvbridge = CvBridge()

        # Subscriber 및 Publisher 생성
        self.create_subscription(CompressedImage, self.image_topic_name, self.image_topic_callback, 10)
        self.create_publisher(Twist, '/drive_cmd_vel', 10)

    def imageCrop(self, _img=np.ndarray(shape=(480, 640))):
        '''
            원하는 이미지 영역 검출
        '''
        return _img[420:480, 0:320]

    def detected(self, _img=np.ndarray(shape=(480, 640))):
        '''
            노란색 차선 영역만 추출
        '''
        hls = cv2.cvtColor(_img, cv2.COLOR_BGR2HLS)
        self.mask_yellow = cv2.inRange(hls, self.YELLOW_LOW_TH, self.YELLOW_HIGH_TH)

        return self.mask_yellow

    def calcLaneDistance(self, _img=np.ndarray(shape=(480, 640))):
        '''
            최종 검출된 이미지를 이용하여 차선의 모멘트 계산
            모멘트의 x, y 좌표 중 차량과의 거리에 해당하는 x를 반환
        '''
        try:
            M = cv2.moments(_img)
            if M['m00'] != 0:  # m00이 0이면 차선이 검출되지 않은 상태이므로 -1 반환
                self.x = int(M['m10'] / M['m00'])
                self.y = int(M['m01'] / M['m00'])
            else:
                self.x = -1
                self.y = -1
        except ZeroDivisionError:
            self.x = -1
            self.y = -1

        print(f"Detected x: {self.x}, y: {self.y}")  # 디버깅 출력
        return float(self.x)  # Int32에서 Float32로 변경

    def visResult(self):
        '''
            최종 결과가 추가된 원본 이미지 (lane_original)
            차선 영역만 ROI로 잘라낸 이미지 (lane_cropped)
            ROI 내부 중 특정 색 영역만 검출한 이미지 (lane_threshold)
        '''
        # 물체 이미지에 표시
        if self.x != -1 and self.y != -1:  # 검출된 물체가 있는 경우에만 표시
            cv2.circle(self.cropped_image, (self.x, self.y), 10, (0, 255, 0), -1)
        
        # 시각화된 결과 출력
        cv2.imshow("lane_original", self.frame)
        cv2.imshow("lane_cropped", self.cropped_image)
        cv2.imshow("lane_thresholded", self.thresholded_detected_image)
        cv2.waitKey(1)

    def reconfigure_callback(self, params):
        '''
            파라미터를 활용하여, 차선 검출을 위한 색 영역 지정
            HLS Color Space를 기반으로 검출
            노란색 및 흰색 차선을 검출을 위한 Threshold 설정
        '''
        for param in params:
            if param.name == "yellow_lane_low":
                self.YELLOW_LOW_TH = np.array([param.value[0], param.value[1], param.value[2]])
            elif param.name == "yellow_lane_high":
                self.YELLOW_HIGH_TH = np.array([param.value[0], param.value[1], param.value[2]])

    def image_topic_callback(self, img):
        '''
            실제 이미지를 입력 받아서 동작하는 부분
            CompressedImage --> OpenCV Type Image 변경 (compressed_imgmsg_to_cv2)
            차선 영역만 ROI 지정 (imageCrop)
            ROI 영역에서 차선 색 영역만 검출 (detected)
            검출된 차선을 기반으로 거리 계산 (calcLaneDistance)
            최종 검출된 값을 기반으로 카메라 좌표계 기준 차선 무게중심 점의 x좌표 Publish
        '''
        self.frame = self.cvbridge.compressed_imgmsg_to_cv2(img, "bgr8")
        self.cropped_image = self.imageCrop(self.frame)
        self.thresholded_detected_image = self.detected(self.cropped_image)
        self.left_distance = self.calcLaneDistance(self.thresholded_detected_image)
        
        # left_distance를 Float32로 퍼블리시
        left_distance_msg = Float32()
        left_distance_msg.data = self.left_distance

        return left_distance_msg.data



def main(args=None):
    rclpy.init(args=args)
    storagy_drive = Storagy_Drive()
    rclpy.spin(storagy_drive)
    storagy_drive.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
