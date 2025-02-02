# %%
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os

# i1 = cv.imread("TP2\\3ASRITP02I01.jpg")
# i2 = cv.imread("TP2\\3ASRITP02I02.jpg")
# i1 = cv.imread("TP2\\TE2_2.jpeg")
# i2 = cv.imread("TP2\\TE3_2.jpg")

# img1 = cv.cvtColor(i1, cv.COLOR_BGR2GRAY)
# img2 = cv.cvtColor(i2, cv.COLOR_BGR2GRAY)

# sift = cv.SIFT_create()

# k_1, des_1 = sift.detectAndCompute(img1, None)
# k_2, des_2 = sift.detectAndCompute(img2, None)

# bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)

# matches = bf.match(des_1, des_2)
# matches = sorted(matches, key=lambda x: x.distance)
# img3 = cv.drawMatches(img1, k_1, img2, k_2, matches[:50], img2, flags=2)


# cv.imshow("Output", img3)
# cv.waitKey(0)
# cv.destroyAllWindows()

# %%
# 2. Points de Moravec

# img = cv.imread(os.path.join("TP2", "3ASRITP02I01.jpg"))
# img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# print(img.shape)


# def moravec(img):
#     h, w = img.shape
#     SM = np.zeros((h, w))
#     window_size = 11
#     half_window = window_size // 2
#     translations = [-5, -3, -1, 1, 3, 5]
#     for i in range(half_window, h - half_window):
#         for j in range(half_window, w - half_window):
#             window = img[
#                 i - half_window : i + half_window + 1,
#                 j - half_window : j + half_window + 1,
#             ]

#             min_diff = float("inf")
#             for a in translations:
#                 for b in translations:
#                     if a == 0 and b == 0:
#                         continue

#                     i_shifted = i + a
#                     j_shifted = j + b

#                     if (
#                         i_shifted - half_window >= 0
#                         and i_shifted + half_window + 1 <= h
#                         and j_shifted - half_window >= 0
#                         and j_shifted + half_window + 1 <= w
#                     ):
#                         window_shifted = img[
#                             i_shifted - half_window : i_shifted + half_window + 1,
#                             j_shifted - half_window : j_shifted + half_window + 1,
#                         ]

#                         diff = np.sum((np.abs(window - window_shifted)))

#                         min_diff = min(min_diff, diff)

#             SM[i, j] = min_diff

#     return SM


# SM = moravec(img)
# plt.imshow(SM, cmap="gray")
# plt.colorbar()
# plt.show()


# n_points = 0
# for i in range(SM.shape[0]):
#     for j in range(SM.shape[1]):
#         if (
#             SM[i, j] > SM[i - 1, j]
#             and SM[i, j] > SM[i + 1, j]
#             and SM[i, j] > SM[i, j - 1]
#             and SM[i, j] > SM[i, j + 1]
#         ):
#             n_points += 1

# print("Nombre de points d'intérêt: ", n_points)


# %%
# LBP


img = cv.imread(os.path.join("TP2", "TP02I01.png"))
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def lbp(img):
    h, w = img.shape
    lbp = np.zeros((h, w))
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            center = img[i, j]
            lbp[i, j] = (
                (img[i - 1, j - 1] > center) << 7
                | (img[i - 1, j] > center) << 6
                | (img[i - 1, j + 1] > center) << 5
                | (img[i, j + 1] > center) << 4
                | (img[i + 1, j + 1] > center) << 3
                | (img[i + 1, j] > center) << 2
                | (img[i + 1, j - 1] > center) << 1
                | (img[i, j - 1] > center)
            )
    return lbp


lbp1 = lbp(img)

n_points = 0
for i in range(lbp1.shape[0]):
    for j in range(lbp1.shape[1]):
        if (
            lbp1[i, j] > lbp1[i - 1, j]
            and lbp1[i, j] > lbp1[i + 1, j]
            and lbp1[i, j] > lbp1[i, j - 1]
            and lbp1[i, j] > lbp1[i, j + 1]
        ):
            n_points += 1

print("Nombre de points d'intérêt: ", n_points)


plt.imshow(lbp1, cmap="gray")
plt.colorbar()
plt.show()
