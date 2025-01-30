from math import cos
from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt
import cv2


def imagesc(img1):
    img2 = (img1 - np.min(img1)) / (np.max(img1) - np.min(img1))
    plt.imshow(img2, cmap="gray")
    plt.show()


hough = cv2.imread("TP3I02.png")
hough_gray = cv2.cvtColor(hough, cv2.COLOR_BGR2GRAY).astype(float)

height, width = 468, 500
thetas = np.linspace(0, np.pi, width)  # Correspond aux colonnes de l'image de Hough
img = np.zeros((height, width))

print(hough_gray.shape)

for x in range(height):
    for y in range(width):
        for theta_index, theta in enumerate(thetas):
            rho = round((x * np.cos(theta) + y * np.sin(theta)))

            if 0 <= rho < hough_gray.shape[0]:
                img[x, y] += hough_gray[int(rho), theta_index]

img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

img = np.log(img + 1)
plt.imshow(img, cmap="gray")
# imagesc(img)
# cv2.waitKey(0)
