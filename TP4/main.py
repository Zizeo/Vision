from copy import deepcopy
from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os


def imagesc(img1):
    img2 = (img1 - np.min(img1)) / (np.max(img1) - np.min(img1))
    plt.imshow(img2, cmap="gray")
    plt.show()


cercle_img = cv.imread(os.path.join("TP4", "cercle.png"))
carre_img = cv.imread(os.path.join("TP4", "carre.png"))
triangle_img = cv.imread(os.path.join("TP4", "triangle.png"))
cercle_img = cv.cvtColor(cercle_img, cv.COLOR_BGR2GRAY).astype(float)
carre_img = cv.cvtColor(carre_img, cv.COLOR_BGR2GRAY).astype(float)
triangle_img = cv.cvtColor(triangle_img, cv.COLOR_BGR2GRAY).astype(float)


cercle_bin = np.where(cercle_img > 68, 0, 1)
carre_bin = np.where(carre_img > 68, 0, 1)
triangle_bin = np.where(triangle_img > 68, 0, 1)

chanfrein_up = np.array([[4, 3, 4], [3, 0, np.inf], [np.inf, np.inf, np.inf]])
chanfrein_low = np.array([[np.inf, np.inf, np.inf], [np.inf, 0, 3], [4, 3, 4]])

chanfrein = np.minimum(chanfrein_up, chanfrein_low)
print(chanfrein)


cercle_pad = np.pad(cercle_bin, 1, mode="constant", constant_values=0)
carre_pad = np.pad(carre_bin, 1, mode="constant", constant_values=0)
triangle_pad = np.pad(triangle_bin, 1, mode="constant", constant_values=0)


def distance_transform(img):
    distance_img = np.full(
        img.shape, np.inf, dtype=float
    )  # Initialize with infinity as float
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            if img[i, j] == 0:
                distance_img[i, j] = np.inf
                # print("inf")
            if img[i, j] == 1:
                distance_img[i, j] = 0
                # print(i, j)

    # imagesc(distance_img)
    # Upper pass
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            for k in range(3):
                for l in range(3):
                    if k == 1 and l == 1:
                        continue
                    ni, nj = i + k - 1, j + l - 1
                    if 0 <= ni < img.shape[0] and 0 <= nj < img.shape[1]:
                        distance_img[i, j] = min(
                            distance_img[i, j],
                            distance_img[ni, nj] + chanfrein_up[k, l],
                        )

    # Lower pass
    for i in range(img.shape[0] - 2, 0, -1):
        for j in range(img.shape[1] - 2, 0, -1):
            for k in range(3):
                for l in range(3):
                    if k == 1 and l == 1:
                        continue
                    ni, nj = i + k - 1, j + l - 1
                    if 0 <= ni < img.shape[0] and 0 <= nj < img.shape[1]:
                        distance_img[i, j] = min(
                            distance_img[i, j],
                            distance_img[ni, nj] + chanfrein_low[k, l],
                        )

    return distance_img


distance_carre = distance_transform(carre_bin)
distance_triangle = distance_transform(triangle_bin)
distance_cercle = distance_transform(cercle_bin)

for i in range(distance_carre.shape[0]):
    for j in range(distance_carre.shape[1]):
        if distance_carre[i, j] == np.inf:
            distance_carre[i, j] = 0
        if distance_triangle[i, j] == np.inf:
            distance_triangle[i, j] = 0
        if distance_cercle[i, j] == np.inf:
            distance_cercle[i, j] = 0

imagesc(distance_carre)
imagesc(distance_triangle)
imagesc(distance_cercle)

cercle2 = cv.imread(os.path.join("TP4", "cercle2.png"))
carre2 = cv.imread(os.path.join("TP4", "carre2.png"))
triangle2 = cv.imread(os.path.join("TP4", "triangle2.png"))
cercle2 = cv.cvtColor(cercle2, cv.COLOR_BGR2GRAY).astype(float)
carre2 = cv.cvtColor(carre2, cv.COLOR_BGR2GRAY).astype(float)
triangle2 = cv.cvtColor(triangle2, cv.COLOR_BGR2GRAY).astype(float)


score_carre = {}
score_triangle = {}
score_cercle = {}

score_carre["carre2"] = float(np.sum(distance_carre * carre2).astype(float))
score_carre["triangle2"] = float(np.sum(distance_carre * triangle2).astype(float))
score_carre["cercle2"] = float(np.sum(distance_carre * cercle2).astype(float))
score_triangle["triangle2"] = float(np.sum(distance_triangle * triangle2).astype(float))
score_triangle["cercle2"] = float(np.sum(distance_triangle * cercle2).astype(float))
score_triangle["carre2"] = float(np.sum(distance_triangle * carre2).astype(float))
score_cercle["carre2"] = float(np.sum(distance_cercle * carre2).astype(float))
score_cercle["triangle2"] = float(np.sum(distance_cercle * triangle2).astype(float))
score_cercle["cercle2"] = float(np.sum(distance_cercle * cercle2).astype(float))

score_carre = dict(sorted(score_carre.items(), key=lambda x: x[1]))
score_triangle = dict(sorted(score_triangle.items(), key=lambda x: x[1]))
score_cercle = dict(sorted(score_cercle.items(), key=lambda x: x[1]))

print("carre: ", score_carre)
print("triangle: ", score_triangle)
print("cercle: ", score_cercle)
