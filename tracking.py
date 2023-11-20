import tkinter as tk
import cv2
import mediapipe as mp
from PIL import Image, ImageTk
import pyautogui

pyautogui.FAILSAFE = False

# setting up tkinker gui 
root = tk.Tk()

root.title("Hansor (Tracking Only)")

title = tk.Label(root, text="Move your hand in front of the camera,", font='Helvetica 10 bold')
title.pack(padx=80, pady=5)
sub_title = tk.Label(root, text="Your pointer finger is now your cursor!\n\n", font='Helvetica 10 bold')
sub_title.pack()
frame_label = tk.Label(root, text="Live View", font='Helvetica 10')
frame_label.pack()
frame_window = tk.Frame(root)
frame_window.pack(padx=10, pady=10)
label = tk.Label(frame_window)
label.pack()
note = tk.Label(root, text="Powered by Google Mediapipe and Open CV - By GG", font="Helvetica 8")
note.pack()

# scale to resize camera capture
frame_scale = 600

# introduce mediapipe utils and detection model
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.85, min_tracking_confidence=0.60) 

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
def update_window():
    success, frame = cap.read()
    if success:

        # process the image to RGB, rescale, and flip
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (frame_scale, h*frame_scale//w))
        frame = cv2.flip(frame, 1)

        # detect hand in image
        frame.flags.writeable = False
        results = hands.process(frame)
        frame.flags.writeable = True

        # draw landmarks if hand detected and move cursor accordingly  
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            
            move_cursor(results.multi_hand_landmarks[0].landmark[8].x, results.multi_hand_landmarks[0].landmark[8].y)

        # convert image array to tkiner image and update window
        photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
        label.config(image=photo)
        
        label.image = photo

    else:
        exit(0)

    label.after(10, update_window)  

# call main loop
update_window()

root.mainloop()
cap.release()