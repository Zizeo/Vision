# %%
# II. Filtrage fréquentiel

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import math

img = cv.imread("TP1/TP01I01.jpg")
if img is None:
    raise Exception("Image introuvable")

img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


f_img = np.fft.fft2(img)
f_img_shift = np.fft.fftshift(f_img)
spectre = 20 * np.log(np.abs(f_img_shift))

plt.subplot(121), plt.imshow(img, cmap="gray")
plt.title("img originale"), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(spectre, cmap="gray")
plt.title("spectre"), plt.xticks([]), plt.yticks([])
plt.show()

if_mg1 = np.fft.ifftshift(f_img_shift)
img1 = np.fft.ifft2(if_mg1)
plt.subplot(121), plt.imshow(img, cmap="gray")
plt.title("img originale"), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(np.abs(img1), cmap="gray")
plt.title("img inverse"), plt.xticks([]), plt.yticks([])
plt.show()

mse = np.mean((img - np.abs(img1) ** 2))
print("MSE: {:.20f}".format(mse))

MSE_tab = []
n = 1
while n < img.shape[0] / 2:
    f_img_shift_tmp = f_img_shift.copy()
    f_img_shift_tmp[0:n, :] = 0
    f_img_shift_tmp[-n:, :] = 0
    f_img_shift_tmp[:, 0:n] = 0
    f_img_shift_tmp[:, -n:] = 0
    if_mg1_tmp = np.fft.ifftshift(f_img_shift_tmp)
    if n == 50:
        img50 = np.fft.ifft2(if_mg1_tmp)
    if n == 100:
        img100 = np.fft.ifft2(if_mg1_tmp)
    if n == 200:
        img200 = np.fft.ifft2(if_mg1_tmp)

    img1_tmp = np.fft.ifft2(if_mg1_tmp)

    mse = np.mean((img - np.abs(img1_tmp)) ** 2)
    MSE_tab.append(mse)
    print("n = {}, MSE: {:.20f}".format(n, mse))
    n += 1

mse_max = max(MSE_tab)
mse_percentage = [mse / mse_max * 100 for mse in MSE_tab]
coeff_percentage = [i / img.shape[0] * 100 for i in range(1, len(MSE_tab) + 1)]
plt.plot(coeff_percentage, mse_percentage)
plt.xlabel("Pourcentage de coefficients mis à 0")
plt.ylabel("Erreur Quadratique Moyenne (%)")
plt.show()

plt.subplot(121), plt.imshow(img, cmap="gray")
plt.title("img originale"), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(np.abs(img50), cmap="gray")
plt.title("img inverse n=50"), plt.xticks([]), plt.yticks([])
plt.show()

plt.subplot(121), plt.imshow(img, cmap="gray")
plt.title("img originale"), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(np.abs(img100), cmap="gray")
plt.title("img inverse n=100"), plt.xticks([]), plt.yticks([])
plt.show()

plt.subplot(121), plt.imshow(img, cmap="gray")
plt.title("img originale"), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(np.abs(img200), cmap="gray")
plt.title("img inverse n=200"), plt.xticks([]), plt.yticks([])
plt.show()

# %%
# III. Caractérisation


nbimg = 50
chemin = "/TP1/bibimage"
signature = np.zeros((nbimg, 18), float)
indx = [0, 50, 75, 100, 125, 150, 200]
indy = [0, 50, 75, 100]
for i in range(nbimg):
    img = plt.imread(
        chemin
        + "\\"
        + str(math.floor((i + 1) / 10))
        + str(math.floor((i + 1) % 10))
        + ".jpg"
    )

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    f_img = np.fft.fft2(img)
    f_img_shift = np.fft.fftshift(f_img)
    spectre = 20 * np.log(np.abs(f_img_shift))
    k = 0
    l = 0
    for j in range(18):
        if k == 6:
            l = l + 1
            k = 0
        signature[i, j] = np.sum(
            np.abs(spectre[indy[l] : indy[l + 1], indx[k] : indx[k + 1]])
        )

print(signature)


num_img = 30
imgchoisie = signature[num_img - 1]

dist = []
for i in range(1, nbimg):
    if i == num_img - 1:
        continue
    dist.append(np.sum(np.abs(imgchoisie - signature[i])))
min_dist = np.min(dist)
index_min_dist = np.argmin(dist)
print("L'image la plus similaire à l'image ", num_img, " est l'image", index_min_dist)
print("La distance minimale est de ", min_dist)


img1 = plt.imread(chemin + "\\" + str(num_img).zfill(2) + ".jpg")
img2 = plt.imread(chemin + "\\" + str(index_min_dist).zfill(2) + ".jpg")

plt.subplot(121), plt.imshow(img1), plt.title("img" + str(num_img))
plt.subplot(122), plt.imshow(img2), plt.title("img" + str(index_min_dist))
plt.show()


# %%
