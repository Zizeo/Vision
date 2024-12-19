from math import cos
from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os


def imagesc(img1):
    img2 = (img1 - np.min(img1)) / (np.max(img1) - np.min(img1))
    plt.imshow(img2, cmap="gray")
    plt.show()


# def get_all_haar_filters():
#     filters = []
#     for i in range(1, 11):
#         for j in range(1, 11):
#             for k in range(11 - i):
#                 for l in range(11 - j):
#                     # motif 1
#                     filter1 = np.zeros((i, j))
#                     filter1[:, : j // 2] = 1
#                     filter1[:, j // 2 :] = -1
#                     filters.append(filter1)
#                     # motif 2
#                     filter2 = np.zeros((i, j))
#                     filter2[: i // 2, :] = 1
#                     filter2[i // 2 :, :] = -1
#                     filters.append(filter2)
#                     # motif 3
#                     filter3 = np.zeros((i, j))
#                     filter3[:, : j // 3] = 1
#                     filter3[:, j // 3 : 2 * j // 3] = -1
#                     filter3[:, 2 * j // 3 :] = 1
#                     filters.append(filter3)
#     return filters

filelist = os.listdir("TP5/training")


def get_haar_filter():
    F = np.zeros((1000, 3301))
    LIGNE = 0
    INDICE = 0
    for fichier in filelist:
        img = cv.imread("TP5/training/" + fichier).astype(float)
        # print(img)
        # Motif 1 (horizontal)
        for H in range(1, 11):
            for L in range(2, 11, 2):
                for X in range(0, 11 - L):
                    for Y in range(0, 11 - H):
                        F[LIGNE, INDICE] += np.sum(img[Y : Y + H, X : X + L // 2])
                        F[LIGNE, INDICE] += -np.sum(img[Y : Y + H, X + L // 2 : X + L])
                        INDICE += 1

        # Motif 2 (vertical)
        for H in range(1, 11):
            for L in range(2, 11, 2):
                for X in range(0, 11 - H):
                    for Y in range(0, 11 - L):
                        F[LIGNE, INDICE] += np.sum(img[Y : Y + L, X : X + H])
                        F[LIGNE, INDICE] += -np.sum(img[Y + L : Y + 2 * L, X : X + H])
                        INDICE += 1

        # Motif 3 (horizontal double)
        for H in range(1, 11):
            for L in range(2, 6, 2):
                for X in range(0, 11 - H):
                    for Y in range(0, 11 - 2 * L):
                        F[LIGNE, INDICE] += np.sum(img[Y : Y + H, X : X + L])
                        F[LIGNE, INDICE] += -np.sum(img[Y : Y + H, X + L : X + 2 * L])
                        F[LIGNE, INDICE] += np.sum(
                            img[Y : Y + H, X + 2 * L : X + 3 * L]
                        )
                        INDICE += 1
        LIGNE = LIGNE + 1
        INDICE = 0
    return F


haar_filters = get_haar_filter()
# print(haar_filters.shape)
np.set_printoptions(threshold=10000)
haar_filters[:500, -1] = 1
haar_filters[500:, -1] = 0
# print(haar_filters)


def sortrows(matrix, columns):
    def custom_key(row):
        return tuple(row[col] for col in columns)

    sorted_matrix = np.array(sorted(matrix, key=custom_key))
    return sorted_matrix


def choixseuil(F, FEATURE):
    # If F contains less than 2 rows, return 0 for threshold and the complete array for FG and FD
    if F.shape[0] < 2:
        return 0, F, F

    # Sort the matrix by the specified feature column
    sorted_matrix = sortrows(F, [FEATURE])

    # Find positions where the ground truth is positive
    pos = np.where(sorted_matrix[:, -1] == 1)[0]

    # If there are no positive examples or all examples are positive
    if len(pos) == 0 or pos[-1] == F.shape[0] - 1:
        seuil = sorted_matrix[0, FEATURE] - 1  # Set threshold below the smallest value
        return seuil, sorted_matrix, sorted_matrix

    # Calculate the optimal threshold
    seuil = (sorted_matrix[pos[-1], FEATURE] + sorted_matrix[pos[-1] + 1, FEATURE]) / 2

    # Create subarrays for FG and FD
    FG = sorted_matrix[sorted_matrix[:, FEATURE] < seuil]
    FD = sorted_matrix[sorted_matrix[:, FEATURE] >= seuil]

    return seuil, FG, FD


FG, FD, seuil = choixseuil(haar_filters, 1)
print(seuil)
# print(FG.shape)
# print(FD.shape)
