import cv2
import numpy as np
import face_recognition as face

cam = cv2.VideoCapture(0)
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

    face_landmarks = face.face_landmarks(small_frame) 
    #gives a list containing a dictionary with features as key, coordinates as values
    print(face_landmarks)

    features = list(face_landmarks[0].values()) #GETTING BUG HERE - LIST INDEX OUT OF RANGE FOR SOME REASON
    for i in features:
        #converting to an numpy array to multiply values by 4 as small_frame is resized to 1/4 of original frame
        #so coordinates in face_landmarks will be 1/4 of the actual coordinates
        #hence we multiply by 4 just like in the case of top,right,bottom,left in face_locations
        k = np.array(i)
        k = k*4
        i = k.tolist()
        for j in i:
            cv2.circle(frame, tuple(j), 1, (0, 0, 255), -1)  #plots small circles in the coordinates (j)  
    cv2.imshow('Face Rec', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cam.release()