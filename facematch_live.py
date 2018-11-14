import face_recognition
import cv2

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

# Load a sample picture and learn how to recognize it.

vish = face_recognition.load_image_file("./sprites/pic2.jpg")
donald = face_recognition.load_image_file("./sprites/donald.jpg")
rihan = face_recognition.load_image_file("./sprites/pic4.jpg")

vish_face_encoding = face_recognition.face_encodings(vish)[0]
donald_face_encoding = face_recognition.face_encodings(donald)[0]
rihan_face_encoding = face_recognition.face_encodings(rihan)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    vish_face_encoding,
    rihan_face_encoding,
    donald_face_encoding
]
known_face_names = [
    "Vishhvak",
    "Rihan",
    "Donald",
]

face_locations = []
face_encodings = []
face_names = []

while True:

    _, frame = cam.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
   
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:

        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        face_names.append(name)


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (255, 0, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting frames
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()