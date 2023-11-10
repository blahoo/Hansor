import tkinter as tk
import cv2
import mediapipe as mp
from PIL import Image, ImageTk
import pyautogui

root = tk.Tk()
root.title("Hansor")

title = tk.Label(root, text="Live Webcam")
title.pack()

frame_scale = 1000

frame_window = tk.Frame(root)
frame_window.pack(padx=10, pady=10)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise Exception("Could not open video device")

label = tk.Label(frame_window)
label.pack()

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def update_frame():

    with mp_hands.Hands(min_detection_confidence=0.85, min_tracking_confidence=0.60) as hands: 

        success, frame = cap.read()
        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (frame_scale, h*frame_scale//w))

            frame = cv2.flip(frame, 1)

            frame.flags.writeable = False
            
            results = hands.process(frame)
            
            frame.flags.writeable = True
                        
            if results.multi_hand_landmarks:
                for num, hand in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4), mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2))

                pyautogui.moveTo(results.multi_hand_landmarks[0].landmark[8].x * pyautogui.size().width, results.multi_hand_landmarks[0].landmark[8].y * pyautogui.size().height)


            photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            label.config(image=photo)
            
            label.image = photo
        label.after(10, update_frame)  
update_frame()

root.mainloop()

cap.release()


