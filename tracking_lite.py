import tkinter as tk
import cv2
import mediapipe as mp
import pyautogui

pyautogui.FAILSAFE = False

# setting up tkinker gui 
root = tk.Tk()
root.title("Hansor_Lite (Tracking Only)")

title = tk.Label(root, text="Move your hand in front of the camera,", font='Helvetica 10 bold')
title.pack(padx=80, pady=5)
sub_title = tk.Label(root, text="Your pointer finger is now your cursor!", font='Helvetica 10 bold')
sub_title.pack()
div = tk.Label(root, text="...")
div.pack(pady=15)
note = tk.Label(root, text="Powered by Google Mediapipe and Open CV - By GG", font="Helvetica 8")
note.pack()


# introduce mediapipe utils and detection model
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(min_detection_confidence=0.80, min_tracking_confidence=0.60) 

# set up camera capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise Exception("Could not open video device")

# scale hand coordinate position and move mouse cursor
def move_cursor(x, y):

    # ([x/y] - min) / (max - min)
    x = (x - 0.2) / (0.8 - 0.2)
    y = (y - 0.2) / (0.5 - 0.2)
    
    # boundry for x and y value
    if x <= 0: x = 0
    if x >= 0.99: x = 0.99
    if y <= 0: y = 0
    if y >= 0.99: y = 0.99

    # print(f"({x}, {y})")

    pyautogui.moveTo(x * pyautogui.size().width, y * pyautogui.size().height)

# main method 
def main_loop():
    success, frame = cap.read()
    if success:

        # process the image to RGB and flip
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)

        # detect hand in image
        frame.flags.writeable = False
        results = hands.process(frame)
        frame.flags.writeable = True

        # draw l    andmarks if hand detected and move cursor accordingly  
        if results.multi_hand_landmarks:            
            move_cursor(results.multi_hand_landmarks[0].landmark[8].x, results.multi_hand_landmarks[0].landmark[8].y)
        
    else:
        cap.release()
        exit(0)
    
    root.after(10, main_loop)  

main_loop()
root.mainloop()
cap.release()