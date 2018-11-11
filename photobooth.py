import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import time
from facefilters import processFrame

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

cvframe = None
what_filter = None


def setFilter(value):
    global what_filter
    what_filter = value


def snapshot():
    global cvframe
    global what_filter
    cv2.imwrite(
        what_filter+"frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cvframe)


def show_frame():
    global cvframe
    _, cvframe = cap.read()
    cvframe = cv2.flip(cvframe, 1)

    processedFrame = processFrame(cvframe, what_filter)

    cv2image = cv2.cvtColor(processedFrame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)


btn1 = tk.Button(window, text="Snapshot", command=snapshot)
btn1.pack(expand=True)

btn2 = tk.Button(window, text="Doggy", command=lambda: setFilter("doggy"))
btn2.pack(expand=True)

btn3 = tk.Button(window, text="Moustache",
                 command=lambda: setFilter("mustache"))
btn3.pack(expand=True)

btn4 = tk.Button(window, text="Find Face",
                 command=lambda: setFilter("findface"))
btn4.pack(expand=True)

show_frame()  # Display 2
window.mainloop()  # Starts GUI
