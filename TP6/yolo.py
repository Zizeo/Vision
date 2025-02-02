# YOLO object detection
import cv2 as cv
import numpy as np
import time
import os

# img = cv.imread("")
# cv.imshow("window", img)
# cv.waitKey(1)

# Load names of classes and get random colors
classes = open("coco.names").read().strip().split("\n")
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")

# Give the configuration and weight files for the model and load the network.
net = cv.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

score_positif = 0
score_negatif = 0
positif_dir1 = os.listdir("Positifs Partie 1")
positif_dir2 = os.listdir("Positifs Partie 2")
negatif_dir1 = os.listdir("Negatifs Partie 1")
negatif_dir2 = os.listdir("Negatifs Partie 2")

positif = []
negatif = []
print("calcul en cours")
for i in range(len(positif_dir1)):
    positif.append(cv.imread("Positifs Partie 1/" + positif_dir1[i]))
    print(i)
for i in range(len(positif_dir2)):
    positif.append(cv.imread("Positifs Partie 2/" + positif_dir2[i]))
    print(i)
for i in range(len(negatif_dir1)):
    negatif.append(cv.imread("Negatifs Partie 1/" + negatif_dir1[i]))
    print(i)
for i in range(len(negatif_dir2)):
    negatif.append(cv.imread("Negatifs Partie 2/" + negatif_dir2[i]))
    print(i)

for img in positif:
# determine the output layer
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the image
    blob = cv.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=False, crop=False)
    r = blob[0, 0, :, :]


    net.setInput(blob)
    # t0 = time.time()
    outputs = net.forward(ln)
    # t = time.time()
    # print("time=", t - t0)

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            if classes[classID] == "person":
                score_positif += 1

for img in negatif:
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the image
    blob = cv.dnn.blobFromImage(img, 1, (416, 416), swapRB=True, crop=False)
    r = blob[0, 0, :, :]

    net.setInput(blob)
    t0 = time.time()
    outputs = net.forward(ln)
    t = time.time()
    print("time=", t - t0)

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            if classes[classID] == "person":
                score_negatif += 1


positif_negatif = len(positif_dir1) + len(positif_dir2) - score_positif
negatif_negatif = len(negatif_dir1) + len(negatif_dir2) - score_negatif


print("accuracy:", (score_positif / (positif_negatif + negatif_negatif)) * 100)
print("recall:", (score_positif / (len(positif_dir1) + len(positif_dir2))) * 100)



# def trackbar2(x):
#     confidence = x / 100
#     r = r0.copy()
#     for output in np.vstack(outputs):
#         if output[4] > confidence:
#             x, y, w, h = output[:4]
#             p0 = int((x - w / 2) * 416), int((y - h / 2) * 416)
#             p1 = int((x + w / 2) * 416), int((y + h / 2) * 416)
#             cv.rectangle(r, p0, p1, 1, 1)
#     cv.imshow("blob", r)
#     text = f"Bbox confidence={confidence}"
#     # cv.displayOverlay('blob', text)


# r0 = blob[0, 0, :, :]
# r = r0.copy()
# cv.imshow("blob", r)
# cv.createTrackbar("confidence", "blob", 50, 101, trackbar2)
# trackbar2(50)

# boxes = []
# confidences = []
# classIDs = []
# h, w = img.shape[:2]

# for output in outputs:
#     for detection in output:
#         scores = detection[5:]
#         classID = np.argmax(scores)
#         confidence = scores[classID]
#         if confidence > 0.5:
#             box = detection[:4] * np.array([w, h, w, h])
#             (centerX, centerY, width, height) = box.astype("int")
#             x = int(centerX - (width / 2))
#             y = int(centerY - (height / 2))
#             box = [x, y, int(width), int(height)]
#             boxes.append(box)
#             confidences.append(float(confidence))
#             classIDs.append(classID)

# indices = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
# if len(indices) > 0:
#     for i in indices.flatten():
#         (x, y) = (boxes[i][0], boxes[i][1])
#         (w, h) = (boxes[i][2], boxes[i][3])
#         color = [int(c) for c in colors[classIDs[i]]]
#         cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
#         text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
#         cv.putText(img, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# cv.imshow("window", img)
# cv.waitKey(0)
# cv.destroyAllWindows()
