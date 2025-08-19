import cv2
import mediapipe as mp
import time
import numpy as np

# MediaPipe 손 인식 초기화
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

# 손 위치 추적용 변수
prev_position = None
still_start_time = None
STILL_THRESHOLD = 20      # 정지 판단 거리 (픽셀)
STILL_TIME_LIMIT = 3      # 정지 지속 시간 (초)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    height, width, _ = frame.shape
    danger_detected = False

    # 기준선 위치 (하단 25%)
    danger_line_y = int(height * 0.75)

    # 기준선 표시 (노란색)
    cv2.line(frame, (0, danger_line_y), (width, danger_line_y), (0, 255, 255), 2)
    cv2.putText(frame, "Danger Zone Below", (10, danger_line_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            wrist = hand_landmarks.landmark[0]  # 손목 좌표
            cx, cy = int(wrist.x * width), int(wrist.y * height)

            # [1] 손이 기준선 아래로 내려갔는지 확인
            if cy > danger_line_y:
                danger_detected = True
                cv2.putText(frame, "Danger: Hand too low!", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # 손목 위치에 빨간 원
                cv2.circle(frame, (cx, cy), 15, (0, 0, 255), -1)

            # [2] 손이 너무 오래 멈췄는지 확인
            if prev_position:
                dist = np.linalg.norm(np.array(prev_position) - np.array((cx, cy)))
                if dist < STILL_THRESHOLD:
                    if still_start_time is None:
                        still_start_time = time.time()
                    elif time.time() - still_start_time > STILL_TIME_LIMIT:
                        danger_detected = True
                        cv2.putText(frame, "Danger: Hand not moving!", (10, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    still_start_time = None  # 움직였으면 초기화
            prev_position = (cx, cy)

            # 손 관절 그리기
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        prev_position = None
        still_start_time = None

    cv2.imshow("Risk Behavior Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()