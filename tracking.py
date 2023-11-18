import tkinter as tk
import cv2
import mediapipe as mp
from PIL import Image, ImageTk
import pyautogui

pyautogui.FAILSAFE = False

root = tk.Tk()
root.title("Hansor")

title = tk.Label(root, text="Move Your Hand!")
title.pack()
frame_window = tk.Frame(root)
frame_window.pack(padx=10, pady=10)
label = tk.Label(frame_window)
label.pack()

frame_scale = 1000

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(min_detection_confidence=0.1, min_tracking_confidence=0.60) 

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise Exception("Could not open video device")

def move_cursor(x, y):

    print(x, y)

    pyautogui.moveTo(x * pyautogui.size().width, y * pyautogui.size().height)


def update_window():
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
                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            
            move_cursor(results.multi_hand_landmarks[0].landmark[8].x, results.multi_hand_landmarks[0].landmark[8].y)

        photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
        label.config(image=photo)
        
        label.image = photo

    else:
        exit(0)

    label.after(10, update_window)  

update_window()

root.mainloop()

cap.release()


