from math import cos
from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


def imagesc(img1):
    img2 = (img1 - np.min(img1)) / (np.max(img1) - np.min(img1))
    plt.imshow(img2, cmap="gray")
    plt.show()


# img = cv.imread("TP3/TP3I01.jpg")
# if img is None:
#     raise Exception("Image introuvable")

# img = cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype(float)

# kernel1 = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]) / 3
# kernel2 = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]) / 3

# filtered1 = cv.filter2D(img, -1, kernel1)
# filtered2 = cv.filter2D(img, -1, kernel2)
# filtered = abs(filtered1) + abs(filtered2)
# max_ = np.max(filtered)
# binarized = np.where(filtered > 10, 255, 0)

# plt.subplot(121), plt.imshow(img, cmap="gray")
# plt.title("img originale"), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(binarized, cmap="gray")
# plt.title("img binarisé"), plt.xticks([]), plt.yticks([])
# plt.show()

# shape_ = binarized.shape
# H = np.zeros((shape_[0] + shape_[1], 360))
# ro = None

# for x in range(shape_[0]):
#     for y in range(shape_[1]):
#         if binarized[x, y] == 255:
#             for t in range(0, 360, 1):
#                 ro = int(x * np.cos(t * np.pi / 180) + y * np.sin(t * np.pi / 180))
#                 if 0 <= ro < shape_[0] + shape_[1]:
#                     H[ro, t] += 1


# imagesc(H)

# lst_x1, lst_y1, lst_x2, lst_y2 = [], [], [], []
# k = 10
# epsilum = 0.01
# indices = np.argsort(H, axis=None)[-k:]
# for index in indices:
#     rho = index // 360
#     theta = index % 360
#     x1, y1, x2, y2 = None, None, None, None
#     for x in range(shape_[0]):
#         for y in range(shape_[1]):
#             if (
#                 binarized[x, y] == 255
#                 and abs(
#                     x * np.cos(theta * np.pi / 180)
#                     + y * np.sin(theta * np.pi / 180)
#                     - rho
#                 )
#                 < epsilum
#             ):
#                 if x1 is None or x < x1:
#                     x1, y1 = x, y
#                 if x2 is None or x > x2:
#                     x2, y2 = x, y
#     if x1 is not None and x2 is not None:
#         print("Point", k, ":", rho, theta, "(", x1, ",", y1, ") (", x2, ",", y2, ")")
#         lst_x1.append(x1)
#         lst_y1.append(y1)
#         lst_x2.append(x2)
#         lst_y2.append(y2)
#     k -= 1


# for x1, y1, x2, y2 in zip(lst_x1, lst_y1, lst_x2, lst_y2):
#     plt.plot([x1, x2], [y1, y2], color="red", linewidth=1)


# plt.show()


img = cv.imread("./TP3/TP3I02.png")
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype(float)
print(img.shape)
img_radon = np.zeros((468, 500))

cosines = [np.cos(t / 100) for t in range(628)]
sines = [np.sin(t / 100) for t in range(628)]

for x in range(468):
    for y in range(500):
        img_radon[x, y] = sum(
            [
                img[int(x * cosines[t] + y * sines[t]), t]
                for t in range(628)
                if 0 <= int(x * cosines[t] + y * sines[t]) < 969
            ]
        )


imagesc(img_radon)
