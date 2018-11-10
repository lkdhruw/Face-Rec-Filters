import cv2
import numpy as np
import face_recognition as face

doggy_nose = cv2.imread("./sprites/doggy_nose.png")
doggy_ears = cv2.imread("./sprites/doggy_ears.png")
doggy_tongue = cv2.imread("./sprites/doggy_tongue.png")
mustache = cv2.imread("./sprites/mustache.png")
rainbow = cv2.imread("./sprites/rainbow.png")
hat = cv2.imread("./sprites/hat.png")
what_filter = "mustache"

# cv2.imshow("g", glasses)


def apply_sprite(sprite, rows, cols):
    # @sprite = sprite to be added
    # rows, cols = Tuples of the form (start, end) = start and end row/col values where sprite is to be masked

    # Segment the part of the image from the main frame
    # Eg: for nose, the whole nose part
    seg = frame[rows[0]*4:rows[1]*4, cols[0]*4: cols[1]*4]
    r, c, _ = seg.shape

    # Resize sprite to fit segment area
    try:
        re_sprite = cv2.resize(sprite, (c, r))
        print(c, r)
    except:
        print("Failed to resize")
        return

    # Thresholding operations using bitwise
    # [refer to bitwise section in this https://docs.opencv.org/3.2.0/d0/d86/tutorial_py_image_arithmetics.html ]

    img2gray = cv2.cvtColor(re_sprite, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(img2gray, 5, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # cv2.imshow("Glass", mask)
    # cv2.waitKey(0)

    img1_bg = cv2.bitwise_and(seg, seg, mask=mask_inv)
    img2_fg = cv2.bitwise_and(re_sprite, re_sprite, mask=mask)
    dst = cv2.add(img1_bg, img2_fg)
    frame[rows[0]*4:rows[1]*4, cols[0]*4: cols[1]*4] = dst


def add_nose_sprite(nose_sprite):
    nose_tip = face_landmarks[0]['nose_tip']
    nose_bridge = face_landmarks[0]['nose_bridge']
    apply_sprite(nose_sprite,
                 (nose_bridge[2][1], nose_tip[0][1]), (nose_tip[0][0], nose_tip[4][0]))


def add_dog_ears():
    left_eyebrow = face_landmarks[0]['left_eyebrow']
    right_eyebrow = face_landmarks[0]['right_eyebrow']
    end_row = left_eyebrow[2][1]
    start_row = end_row - 10
    start_col = left_eyebrow[0][0]
    end_col = right_eyebrow[4][0]
    apply_sprite(doggy_ears, (start_row, end_row), (start_col, end_col))


def add_hat():
    left_eyebrow = face_landmarks[0]['left_eyebrow']
    right_eyebrow = face_landmarks[0]['right_eyebrow']
    eyebrows = left_eyebrow + right_eyebrow
    x_values = [tup[0] for tup in eyebrows]
    y_values = [tup[1] for tup in eyebrows]
    start_row = min(y_values) - 25
    end_row = min(y_values) - 5
    start_col = min(x_values) - 15
    end_col = max(x_values) + 15
    apply_sprite(hat, (start_row, end_row), (start_col, end_col))


def add_mustache():
    upper_lip = face_landmarks[0]["top_lip"]
    x_values = [tup[0] for tup in upper_lip]
    y_values = [tup[1] for tup in upper_lip]
    start_row = min(y_values) - 6
    end_row = max(y_values) - 3
    start_col = min(x_values) - 2
    end_col = max(x_values) + 2

    apply_sprite(mustache, (start_row, end_row), (start_col, end_col))


def apply_filters():
    if what_filter == "doggy":
        add_nose_sprite(doggy_nose)
        add_dog_ears()
    elif what_filter == "mustache":
        add_mustache()
        add_hat()


cam = cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)

while True:

    _, frame = cam.read()
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
        apply_filters()

    cv2.imshow("Frame", frame)

    pressed_key = cv2.waitKey(1) & 0xFF
    if pressed_key == ord('q'):
        break
    elif pressed_key == ord('m'):
        what_filter = "mustache"
    elif pressed_key == ord('d'):
        what_filter = "doggy"

print(face_landmarks[0]['bottom_lip'])
cv2.destroyAllWindows()
