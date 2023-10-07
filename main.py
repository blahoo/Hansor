import mediapipe as mp
import cv2
import time

cap = cv2.VideoCapture(0)

while True:
    success, image = cap.read()
    cv2.imshow("Hand Tracking", image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break;

cap.release()
cv2.destroyAllWindows()