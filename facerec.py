import cv2
import numpy as np
import face_recognition as face

cam = cv2.VideoCapture(0)

doggy_tongue = cv2.imread("./sprites/doggy_tongue.png")

while True:

    _, frame = cam.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    face_locations = face.face_locations(small_frame, model='hog')
    print(face_locations)

    for top, right, bottom, left in face_locations:
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        print(top, right, bottom, left)
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

    face_landmarks = []
    # Do face facial recognition only if a face is detected
    if len(face_locations):
        face_landmarks = face.face_landmarks(small_frame)
    # gives a list containing a dictionary with features as key, coordinates as values
    print(face_landmarks)

    # Check if face_landmarks[0] is present
    # If it is present, highlight the face landmarks
    # Else just pass silently
    try:
        features = list(face_landmarks[0].values())
    except:
        pass
    else:
        for i in features:
            # converting to an numpy array to multiply values by 4 as small_frame is resized to 1/4 of original frame
            # so coordinates in face_landmarks will be 1/4 of the actual coordinates
            # hence we multiply by 4 just like in the case of top,right,bottom,left in face_locations
            k = np.array(i)
            k = k*4
            i = k.tolist()
            for j in i:
                # plots small circles in the coordinates (j)
                # cv2.circle(frame, tuple(j), 1, (0, 0, 255), -1)
                # print(face_landmarks[0]['top_lip'])
                top_lip = face_landmarks[0]['top_lip']
                bottom_lip = face_landmarks[0]['bottom_lip']
                frame[top_lip[0]:bottom_lip[0], top_lip[1]:bottom_lip[0]] = doggy_tongue

    cv2.imshow('Face Rec', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cam.release()
