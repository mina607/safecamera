from PIL import ImageFont, ImageDraw, Image
import numpy as np
import cv2
import mediapipe as mp

# 미디어파이프 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# 한글 폰트 경로 (Windows 기준)
font_path = "C:/Windows/Fonts/malgun.ttf"  # 또는 원하는 다른 ttf 경로

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    height, width, _ = frame.shape
    danger_detected = False

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            min_y = min(lm.y for lm in hand_landmarks.landmark) * height

            if min_y < height / 3:
                danger_detected = True

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 기준선 표시
    cv2.line(frame, (0, int(height / 3)), (width, int(height / 3)), (0, 255, 255), 2)

    # PIL로 변환 후 한글 텍스트 출력
    if danger_detected:
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)
        font = ImageFont.truetype(font_path, 40)

        draw.text((50, 50), "위험 행동 감지!", font=font, fill=(255, 0, 0))  # 빨간색

        # 다시 OpenCV용 이미지로 변환
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        print('\a')  # 비프음

    cv2.imshow("Hand Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()