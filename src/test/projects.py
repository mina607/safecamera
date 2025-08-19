import cv2
import numpy as np

# 템플릿 이미지 3개 불러오기
template1 = cv2.imread('../../../img/capture1.png', 0)
if template1 is None:
    print("capture1.png 파일을 읽을 수 없습니다.")
template2 = cv2.imread('../../../img/capture2.png', 0)
if template2 is None:
    print("capture2.png 파일을 읽을 수 없습니다.")
template3 = cv2.imread('../../../img/capture3.png', 0)
if template3 is None:
    print("capture3.png 파일을 읽을 수 없습니다.")

templates = [template1, template2, template3]
threshold = 0.7

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        # 검은 화면 만들기
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # 메시지 띄우기
        cv2.putText(frame, "No camera detected", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Risk Detection', frame)

        # q 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # 카메라가 안 잡히면 계속 여기서 대기
        continue

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    danger_detected = False

    for template in templates:
        if template is None:
            continue

        w, h = template.shape[::-1]
        res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)

        if len(loc[0]) > 0:
            danger_detected = True
            pt = (loc[1][0], loc[0][0])
            cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
            break

    if danger_detected:
        cv2.putText(frame, "🚨 위험 행동 감지!", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Risk Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()