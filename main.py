import tkinter as tk
import cv2
import mediapipe as mp
from PIL import Image, ImageTk
import pyautogui
import time

# set up tkinter gui
root = tk.Tk()
root.title("Hansor")

title = tk.Label(root, text="Live Webcam")
title.pack()

frame_scale = 1000

frame_window = tk.Frame(root)
frame_window.pack(padx=10, pady=10)

live_window = tk.Label(frame_window)
live_window.pack()

# set up video webcame capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise Exception("Could not open video device")

# set up mediapip utils
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
BaseOptions = mp.tasks.BaseOptions
RunningMode = mp.tasks.vision.RunningMode
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult

# min and max confidence
detection_confidence_level = 0.8
tracking_confidence_level = 0.5

# function to handle model output
last_action = time.time()

def result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global last_action

    try:
        #print(result.gestures[0][0].category_name, result.hand_landmarks[0][8].x, result.hand_landmarks[0][8].y)
        #print(time.time())

        if result.gestures[0][0].category_name == "Pointing_Up":
            move_cursor(result.hand_landmarks[0][8].x, result.hand_landmarks[0][8].y)

        elif result.gestures[0][0].category_name == "Victory":
            if last_action - time.time() >= 1:
                return
            left_click()
    
    finally:
        print(Exception)
        return


def left_click():
    pyautogui.leftClick(pyautogui.position())

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


# import model and define mediapipe configuration
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='model.task'),
    running_mode=RunningMode.LIVE_STREAM,
    num_hands = 1,
    min_hand_detection_confidence=detection_confidence_level,
    min_tracking_confidence=tracking_confidence_level,
    result_callback=result)

with mp.tasks.vision.GestureRecognizer.create_from_options(options) as recog:

    stamp = 0

    def main_loop():
        _, frame = cap.read()

        # preprocess image
        image = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1)
        mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = image)

        global stamp
        stamp +=1

        # asynchronously process image
        recog.recognize_async(mp_image, stamp)

        # convert image to tkinter image and update live feed frame
        photo = ImageTk.PhotoImage(image=Image.fromarray(image))
        live_window.config(image=photo)

        live_window.image = photo
        live_window.after(100, main_loop)  

    main_loop()
    root.mainloop()
    cap.release()


