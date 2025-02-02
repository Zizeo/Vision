import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import argparse


def detectAndDisplay(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    # -- Detect faces
    # faces = face_cascade.detectMultiScale(frame_gray)
    # for x, y, w, h in faces:
    #     center = (x + w // 2, y + h // 2)
    #     frame = cv.ellipse(frame, center, (w // 2, h // 2), 0, 0, 360, (255, 0, 255), 4)
    #     faceROI = frame_gray[y : y + h, x : x + w]
        # -- In each face, detect eyes
        # eyes = eyes_cascade.detectMultiScale(faceROI)
        # for x2, y2, w2, h2 in eyes:
        #     eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
        #     radius = int(round((w2 + h2) * 0.25))
        #     frame = cv.circle(frame, eye_center, radius, (255, 0, 0), 4)
    bodies = body_cascade.detectMultiScale(frame_gray)
    if len(bodies) != 0:
        bool_body = True
    for x3, y3, w3, h3 in bodies:
        body_center = (x3 + w3 // 2, y3 + h3 // 2)
        radius = int(round((w3 + h3) * 1))
        frame = cv.circle(frame, body_center, radius, (0, 255, 0), 4)
    cv.imshow("Capture - Face detection", frame)


parser = argparse.ArgumentParser(description="Code for Cascade Classifier tutorial.")
parser.add_argument(
    "--face_cascade",
    help="Path to face cascade.",
    default="face.xml",
)
parser.add_argument(
    "--eyes_cascade",
    help="Path to eyes cascade.",
    default="eyes.xml",
)
parser.add_argument(
    "--body_cascade",
    help="Path to body cascade.",
    default="body.xml",
)

parser.add_argument("--camera", help="Camera divide number.", type=int, default=0)
args = parser.parse_args()
face_cascade_name = args.face_cascade
eyes_cascade_name = args.eyes_cascade
body_cascade_name = args.body_cascade

face_cascade = cv.CascadeClassifier()
eyes_cascade = cv.CascadeClassifier()
body_cascade = cv.CascadeClassifier()
# -- 1. Load the cascades
# -- 1. Load the cascades
if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
    print("--(!)Error loading face cascade")
    exit(0)
if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_name)):
    print("--(!)Error loading eyes cascade")
    exit(0)
if not body_cascade.load(cv.samples.findFile(body_cascade_name)):
    print("--(!)Error loading body cascade")
    exit(0)
camera_device = args.camera
# -- 2. Read the video stream
# cap = cv.VideoCapture(camera_device)
# if not cap.isOpened:
#     print("--(!)Error opening video capture")
#     exit(0)
# while True:
#     ret, frame = cap.read()
#     if frame is None:
#         print("--(!) No captured frame -- Break!")
#         break
#     detectAndDisplay(frame)
#     if cv.waitKey(10) == 27:
#         break

img = cv.imread("Tests_partie1.jpg")
detectAndDisplay(img)
cv.waitKey(0)
