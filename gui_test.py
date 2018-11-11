import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import time

# Set up GUI
window = tk.Tk()
window.wm_title("Filtero")
window.config(background="#ffffff")

# Graphics window
imageFrame = tk.Frame(window, width=600, height=500)
imageFrame.pack()

# Capture video frames
lmain = tk.Label(imageFrame)
lmain.grid(row=0, column=0)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

frame = None


def snapshot():
    global frame
    cv2.imwrite(
        "frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", frame)


def show_frame():
    global frame
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)


btn1 = tk.Button(window, text="Snapshot", command=snapshot)
btn1.pack(anchor=tk.CENTER, expand=True)

show_frame()  # Display 2
window.mainloop()  # Starts GUI
