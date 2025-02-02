from math import cos
from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
from functools import lru_cache
import json


def imagesc(img1):
    img2 = (img1 - np.min(img1)) / (np.max(img1) - np.min(img1))
    plt.imshow(img2, cmap="gray")
    plt.show()


filelist = os.listdir(os.path.join("TP5", "training"))


def get_haar_filter():
    F = np.zeros((1000, 3301))
    LIGNE = 0
    INDICE = 0
    for fichier in filelist:
        img = cv.imread(os.path.join("TP5", "training", fichier)).astype(float)
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
print(haar_filters)
print(haar_filters.shape)


def sortrows(matrix, columns):
    def custom_key(row):
        return tuple(row[col] for col in columns)

    sorted_matrix = np.array(sorted(matrix, key=custom_key))
    return sorted_matrix



def choixseuil(F, FEATURE):
    # If F contains less than 2 rows, return 0 for threshold and the complete array for FG and FD
    if F.shape[0] < 2:
        return 0, F, F

    sorted_matrix = sortrows(F, FEATURE)

    pos = np.where(sorted_matrix[:, -1] == 1)[0]

    if len(pos) == 0 or pos[-1] == F.shape[0] - 1:
        seuil = sorted_matrix[0, FEATURE] - 1
        return seuil, sorted_matrix, sorted_matrix

    seuil = (sorted_matrix[pos[-1], FEATURE] + sorted_matrix[pos[-1] + 1, FEATURE]) / 2

    FG = sorted_matrix[sorted_matrix[:, FEATURE] < seuil]
    FD = sorted_matrix[sorted_matrix[:, FEATURE] >= seuil]

    return seuil, FG, FD


# Example dataset
F = np.array(
    [
        [0.1, 1],  # Negative example
        [0.4, 1],  # Positive example
        [0.5, 1],  # Positive example
        [0.6, 0],  # Negative example
        [0.8, 1],  # Positive example
        [0.9, 0],  # Negative example
    ]
)

FEATURE = [1]

seuil, FG, FD = choixseuil(F, FEATURE)

print("Seuil Optimale:", seuil)
print("FG (Below Threshold):")
print(FG)
print("FD (At or Above Threshold):")
print(FD)


# NBARBRES = 10
# NBFEUILLE = 10
# NBLEVELS = 4
# FORET = np.zeros((NBARBRES, NBFEUILLE, NBLEVELS, 2))


# FEATURE = 1


# # @lru_cache(maxsize=None)
# def construire_arbre(F_tuple, feature, level):
#     print(level)
#     if level >= NBLEVELS:
#         return None
#     F = np.array(F_tuple)  # Convert back to numpy array for processing
#     if len(F) == 0:
#         level += 1
#         return None
#     seuil, FG, FD = choixseuil(F, feature)
#     noeud = {"seuil": seuil, "feature": feature, "FG": FG, "FD": FD}

#     if len(FG) == 0:
#         level += 1
#         noeud["FG"] = None
#     else:
#         level += 1
#         noeud["FG"] = construire_arbre(
#             tuple(FG), feature, level
#         )  # Convert to tuple for caching
#     if len(FD) == 0:
#         level += 1
#         noeud["FD"] = None
#     else:
#         level += 1
#         noeud["FD"] = construire_arbre(
#             tuple(FD), feature, level
#         )  # Convert to tuple for caching
#     level += 1
#     return noeud


# FORET = []
# for i in range(NBARBRES):
#     for j in range(NBFEUILLE):
#         level = 0
#         FORET.append(construire_arbre(tuple(F), FEATURE, level))

# print(json.dumps(FORET, indent=4, separators=(',', ': ')))
# print(len(FORET))