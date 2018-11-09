import cv2
import numpy as np
import face_recognition as face

doggy_nose = cv2.imread("./sprites/doggy_nose.png")
mustache = cv2.imread("./sprites/mustache.png")
nose_sprite = doggy_nose


def apply_sprite(sprite, rows, cols):
    # @sprite = sprite to be added
    # rows, cols = Tuples of the form (start, end) = start and end row/col values where sprite is to be masked

    # Segment the part of the image from the main frame
    # Eg: for nose, the whole nose part
    seg = frame[rows[0]*4:rows[1]*4, cols[0]*4: cols[1]*4]
    r, c, _ = seg.shape

    # Resize sprite to fit segment area
    re_sprite = cv2.resize(sprite, (c, r))

    # Thresholding operations using bitwise
    # [refer to bitwise section in this https://docs.opencv.org/3.2.0/d0/d86/tutorial_py_image_arithmetics.html ]

    img2gray = cv2.cvtColor(re_sprite, cv2.COLOR_BGR2GRAY)

    ret, mask = cv2.threshold(img2gray, 5, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    img1_bg = cv2.bitwise_and(seg, seg, mask=mask_inv)
    img2_fg = cv2.bitwise_and(re_sprite, re_sprite, mask=mask)
    dst = cv2.add(img1_bg, img2_fg)
    frame[rows[0]*4:rows[1]*4, cols[0]*4: cols[1]*4] = dst


def add_nose_sprite(nose_sprite):
    nose_tip = face_landmarks[0]['nose_tip']
    nose_bridge = face_landmarks[0]['nose_bridge']
    apply_sprite(nose_sprite,
                 (nose_bridge[2][1], nose_tip[0][1]), (nose_tip[0][0], nose_tip[4][0]))


cam = cv2.VideoCapture(0)

while True:

    _, frame = cam.read()
    print("Live")
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    face_locations = face.face_locations(small_frame, model='hog')

    for top, right, bottom, left in face_locations:
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

    face_landmarks = []

    if len(face_locations):
        face_landmarks = face.face_landmarks(small_frame)
        add_nose_sprite(nose_sprite)

    cv2.imshow("Frame", frame)

    pressed_key = cv2.waitKey(1) & 0xFF
    if pressed_key == ord('q'):
        break
    elif pressed_key == ord('m'):
        nose_sprite = mustache
    elif pressed_key == ord('d'):
        nose_sprite = doggy_nose

cv2.destroyAllWindows()
