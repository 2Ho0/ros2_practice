import cv2
from picamera2 import Picamera2
import time
import pigpio

# 핀 설정
DIR1, PWM1 = 17, 18
DIR2, PWM2 = 22, 23
pi = pigpio.pi()
for pin in [DIR1, PWM1, DIR2, PWM2]:
    pi.set_mode(pin, pigpio.OUTPUT)
    pi.set_PWM_dutycycle(pin, 0)

# 카메라 설정
cam = Picamera2()
cam.preview_configuration.main.size = (640, 480)
cam.preview_configuration.main.format = "RGB888"
cam.configure("preview")
cam.start()
time.sleep(1)

# ArUco 마커 설정
aruco = cv2.aruco
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
aruco_params = aruco.DetectorParameters()

# 상태 변수
last_seen_time = time.time()
searching = False
last_turn = None # 'left', 'right', or None
reverse_timer = None

print("마커 추적 시작 (q 누르면 종료)")

try:
    while True:
        frame = cam.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
        now = time.time()

        if ids is not None:
            c = corners[0][0]
            cx = int((c[0][0] + c[2][0]) / 2)
            area = cv2.contourArea(c)
            screen_center = frame.shape[1] // 2
            last_seen_time = now
            searching = False
            reverse_timer = None

            if area >= 60000:
                print("❗ 너무 가까움 → 정지")
                pi.set_PWM_dutycycle(PWM1, 0)
                pi.set_PWM_dutycycle(PWM2, 0)
                last_turn = None

            elif cx < screen_center - 135:
                print("↺ 좌회전")
                last_turn = "left"
                pi.write(DIR1, 1)
                pi.write(DIR2, 1)
                pi.set_PWM_dutycycle(PWM1, 25)
                pi.set_PWM_dutycycle(PWM2, 60)

            elif cx > screen_center + 135:
                print("↻ 우회전")
                last_turn = "right"
                pi.write(DIR1, 1)
                pi.write(DIR2, 1)
                pi.set_PWM_dutycycle(PWM1, 60)
                pi.set_PWM_dutycycle(PWM2, 25)

            else:
                print("↑ 전진")
                last_turn = None
                pi.write(DIR1, 1)
                pi.write(DIR2, 1)
                pi.set_PWM_dutycycle(PWM1, 70)
                pi.set_PWM_dutycycle(PWM2, 70)

        else:
            time_missing = now - last_seen_time

            if time_missing > 20:
                print("🛑 20초 동안 마커 없음 → 프로그램 자동 종료")
                break

            if time_missing > 1 and not searching:
                print("🔍 마커 사라짐 → 제자리 회전 시작")
                searching = True

                if last_turn == "left":
                    pi.write(DIR1, 0)
                    pi.write(DIR2, 1)
                elif last_turn == "right":
                    pi.write(DIR1, 1)
                    pi.write(DIR2, 0)
                else:
                    pi.write(DIR1, 0)
                    pi.write(DIR2, 1)

                pi.set_PWM_dutycycle(PWM1, 30)
                pi.set_PWM_dutycycle(PWM2, 30)
                search_start_time = now

            if searching and ids is not None and reverse_timer is None:
                print("✅ 마커 재발견 → 반대 방향 0.3초")
                reverse_timer = now
                if last_turn == "left":
                    pi.write(DIR1, 1)
                    pi.write(DIR2, 0)
                elif last_turn == "right":
                    pi.write(DIR1, 0)
                    pi.write(DIR2, 1)
                else:
                    pi.write(DIR1, 0)
                    pi.write(DIR2, 1)

                pi.set_PWM_dutycycle(PWM1, 30)
                pi.set_PWM_dutycycle(PWM2, 30)

            if reverse_timer and now - reverse_timer >= 0.3:
                reverse_timer = None
                searching = False

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    pi.set_PWM_dutycycle(PWM1, 0)
    pi.set_PWM_dutycycle(PWM2, 0)
    pi.stop()
    cam.stop()
    print("✅ 프로그램 정상 종료")