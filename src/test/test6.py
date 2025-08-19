import cv2
import numpy as np

# 흰색 배경의 검은 사각형 이미지를 생성
img = np.ones((300, 300, 3), dtype=np.uint8) * 255
cv2.rectangle(img, (50, 50), (250, 250), (0, 0, 0), 3)

# 이미지 보여주기
try:
    cv2.imshow("Test Window", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("✅ cv2.imshow() 동작 정상입니다.")
except cv2.error as e:
    print("❌ cv2.imshow() 동작 실패")
    print(e)
