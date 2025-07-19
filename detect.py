#! /usr/bin/python

from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import imutils
import pickle
import time
import cv2
from picamera2 import Picamera2
from gpiozero import Servo
from time import sleep

# Not rlly using servo pitch tho
servoPitch = Servo(24)
servoYaw = Servo(25)
currentname = "unknown"

# Classify diff faces from encodings.pickle file created from train_model.py
encodingsP = "encodings.pickle"

# Load known faces and embeddings along with OpenCV's Haar
# cascade for face detection
print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(encodingsP, "rb").read())

# Initialize the camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()
#time.sleep(2.0)

# Start FPS counter
fps = FPS().start()

yaw_position = 1.0
increment = -0.05  # Small increments so smoother movement, still has seizure tho

# Update servo angle
def update_servo_position():
    global yaw_position, increment
    if yaw_position <= -1.0 or yaw_position >= 1.0:
        increment = -increment  # Reverse direction
    yaw_position += increment
    yaw_position = max(-1.0, min(1.0, yaw_position))  # Ensure within range
    servoYaw.value = yaw_position
    sleep(0.1)

# Loop until face detected
face_detected = False
while not face_detected:
    update_servo_position()
    
    # Grab frame from camera
    frame = picam2.capture_array()
    frame = imutils.resize(frame, width=500)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Facial bounding boxes
    boxes = face_recognition.face_locations(frame)
    encodings = face_recognition.face_encodings(frame, boxes)
    names = []

    # Loop over facial encodings/detections
    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"

        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            name = max(counts, key=counts.get)

            if currentname != name:
                currentname = name
                print(currentname)

        names.append(name)

    if names:  # If any name is detected
        face_detected = True
        servoYaw.value = 0  # Stop servo

    cv2.imshow("Facial Recognition is Running", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# Main loop
while True:
    frame = picam2.capture_array()
    frame = imutils.resize(frame, width=500)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(frame)
    encodings = face_recognition.face_encodings(frame, boxes)
    names = []

    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"

        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            name = max(counts, key=counts.get)

            if currentname != name:
                currentname = name
                print(currentname)

        names.append(name)

    for ((top, right, bottom, left), name) in zip(boxes, names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 225), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 255), 2)

    if "Jonathan" in names:
        print("Access granted! :>")

    if "Mom" in names:
        print("GO AWAY >:(")

    if "Finkelstein" in names:
        print("YAY :D")

    cv2.imshow("Facial Recognition is Running", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    fps.update()

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
picam2.stop()
