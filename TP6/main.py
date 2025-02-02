import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import argparse
import signal
import sys
import os
import json

VIDEO = False
CALCULE = False


def signal_handler(sig, frame):
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def detectAndDisplay(frame):
    bool_face = False
    bool_body = False
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    # -- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    if len(faces) != 0:
        bool_face = True
    for x, y, w, h in faces:
        center = (x + w // 2, y + h // 2)
        frame = cv.ellipse(frame, center, (w // 2, h // 2), 0, 0, 360, (255, 0, 255), 4)
        faceROI = frame_gray[y : y + h, x : x + w]
        # -- In each face, detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI)
        for x2, y2, w2, h2 in eyes:
            eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
            radius = int(round((w2 + h2) * 0.25))
            frame = cv.circle(frame, eye_center, radius, (255, 0, 0), 4)
    bodies = body_cascade.detectMultiScale(frame_gray)
    if len(bodies) != 0:
        bool_body = True
    for x3, y3, w3, h3 in bodies:
        body_center = (x3 + w3 // 2, y3 + h3 // 2)
        radius = int(round((w3 + h3) * 1))
        frame = cv.circle(frame, body_center, radius, (0, 255, 0), 4)

    if bool_face:  # or bool_body:
        return 1
    else:
        return 0
    # cv.imshow("Capture - Face detection", frame)


parser = argparse.ArgumentParser(description="Code for Cascade Classifier tutorial.")
parser.add_argument(
    "--face_cascade",
    help="Path to face cascade.",
    default="TP6/face.xml",
)
parser.add_argument(
    "--eyes_cascade",
    help="Path to eyes cascade.",
    default="TP6/eyes.xml",
)
parser.add_argument(
    "--body_cascade",
    help="Path to body cascade.",
    default="TP6/body.xml",
)

parser.add_argument("--camera", help="Camera divide number.", type=int, default=0)
args = parser.parse_args()
face_cascade_name = args.face_cascade
eyes_cascade_name = args.eyes_cascade
body_cascade_name = args.body_cascade
face_cascade = cv.CascadeClassifier()
body_cascade = cv.CascadeClassifier()
eyes_cascade = cv.CascadeClassifier()

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

if VIDEO:
    cap = cv.VideoCapture(camera_device)
    if not cap.isOpened:
        print("--(!)Error opening video capture")
        exit(0)
    while True:
        ret, frame = cap.read()
        if frame is None:
            print("--(!) No captured frame -- Break!")
            break
        detectAndDisplay(frame)
        if cv.waitKey(10) == 27:
            break
else:
    while True:
        img = cv.imread("TP6/Tests_partie1.jpg")
        detectAndDisplay(img)
        if cv.waitKey(0) == 27:
            cv.destroyAllWindows()
            break

# positif_dir1 = os.listdir("TP6/Positifs Partie 1")
# positif_dir2 = os.listdir("TP6/Positifs Partie 2")
# negatif_dir1 = os.listdir("TP6/Negatifs Partie 1")
# negatif_dir2 = os.listdir("TP6/Negatifs Partie 2")

# if CALCULE:
#     positif = []
#     negatif = []
#     print("calcul en cours")
#     for i in range(len(positif_dir1)):
#         positif.append(cv.imread("TP6/Positifs Partie 1/" + positif_dir1[i]))
#         print(i)
#     for i in range(len(positif_dir2)):
#         positif.append(cv.imread("TP6/Positifs Partie 2/" + positif_dir2[i]))
#         print(i)
#     for i in range(len(negatif_dir1)):
#         negatif.append(cv.imread("TP6/Negatifs Partie 1/" + negatif_dir1[i]))
#         print(i)
#     for i in range(len(negatif_dir2)):
#         negatif.append(cv.imread("TP6/Negatifs Partie 2/" + negatif_dir2[i]))
#         print(i)

#     positif_score = 0
#     negatif_score = 0
#     for i in range(len(positif)):
#         print(i)
#         positif_score += detectAndDisplay(positif[i])
#     for i in range(len(negatif)):
#         print(i)
#         negatif_score += detectAndDisplay(negatif[i])


# # positif_score = 261
# # negatif_score = 121
# # print(positif_score)
# # print(negatif_score)
# # positif_negatif = len(positif_dir1) + len(positif_dir2) - positif_score
# # negatif_negatif = len(negatif_dir1) + len(negatif_dir2) - negatif_score


# # print("accuracy:", (positif_score / (positif_score + negatif_score)) * 100)
# # print("recall:", (positif_score / (len(positif_dir1) + len(positif_dir2))) * 100)

# accuracies = []
# stage = []
# with open("TP6/result.json") as f:
#     json_data = json.load(f)
#     i = 0
#     items = [item for item in json_data]
#     for item in items[1:]:
#         accuracies.append(json_data[item]["accuracy"])
#         print(item)
#         stage.append(20 - i)
#         i += 1


# # les niveaux de cascade permettent de mieux de filtrer la detection
# # augmantant le nombre de vrai positif, augmantant la precision

# # pour avoir une bonne estimation de la qualité du modèle il faut coupler la precision
# # au recall -> F1 score

# plt.plot(stage, accuracies)
# plt.xlabel("stage")
# plt.ylabel("accuracy")
# plt.show()
