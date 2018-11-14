import cv2
import numpy as np
import face_recognition

import tkinter as tk
from PIL import ImageTk, Image


#"./sprites/donald.jpg", "./sprites/filter.jpg")

known_image = face_recognition.load_image_file("./sprites/donald.jpg")
unknown_image = face_recognition.load_image_file("./sprites/filter.jpg")

known_encoding = face_recognition.face_encodings(known_image)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

results = face_recognition.compare_faces([known_encoding], unknown_encoding)
print(results)