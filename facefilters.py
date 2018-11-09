import cv2
import numpy as np
import face_recognition as face


doggy_nose = cv2.imread("./sprites/doggy_nose.png")
frame = cv2.imread("./sprites/Pic.jpg")
small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

face_locations = face.face_locations(small_frame, model='hog')

for top, right, bottom, left in face_locations:
    top *= 4
    right *= 4
    bottom *= 4
    left *= 4
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

face_landmarks = []
# Do face facial recognition only if a face is detected
if len(face_locations):
    face_landmarks = face.face_landmarks(small_frame)
# gives a list containing a dictionary with features as key, coordinates as values
#print(face_landmarks)
print(face_landmarks[0]['nose_tip']);
print(face_landmarks[0]['nose_bridge']);
seg = frame[69*4:80*4, 123*4:140*4]
rows, cols, s = seg.shape
nose = cv2.resize(doggy_nose, (cols,rows));

img2gray = cv2.cvtColor(nose,cv2.COLOR_BGR2GRAY)

ret, mask = cv2.threshold(img2gray, 5, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

img1_bg = cv2.bitwise_and(seg, seg, mask = mask_inv)
img2_fg = cv2.bitwise_and(nose,nose,mask = mask)
dst = cv2.add(img1_bg,img2_fg)
frame[69*4:80*4, 123*4:140*4] = dst

cv2.imshow('Face Rec', frame)
cv2.imwrite( "filter.jpg", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
