import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("../img/edificio.jpg")
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

img = cv.resize(img, dsize=None, fx=0.3, fy=0.3)
cv.imshow('img', img)

# Detección de boprdes con gradiente con filtro pewwitt

Hx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
Gx = cv.filter2D(np.double(img), -1, Hx)
Gx_img = Gx.copy()
min = np.min(Gx_img)
Gx_img -= min
max = np.max(Gx_img)
Gx_img = np.uint8((Gx_img / max) * 255)
cv.imshow('Gx_img', Gx_img)

Hy = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
Gy = cv.filter2D(np.double(img), -1, Hy)
Gy_img = Gy.copy()
min_y = np.min(Gy_img)
Gy_img -= min_y
max_y = np.max(Gy_img)
Gy_img = np.uint8((Gy_img / max_y) * 255)
cv.imshow('Gy_img', Gy_img)

# Calcular el gradiente total
G = np.sqrt(Gx**2 + Gy**2)

# Normalizar el gradiente total para mostrarlo correctamente
G -= np.min(G)
G /= np.max(G)
G *= 255
G = np.uint8(G)

cv.imshow('Gradiente Total', G)

plt.hist(G.flatten(), bins=256)
plt.show()

_,G_binary = cv.threshold(G, 25, 255, cv.THRESH_BINARY)
cv.imshow('G_bin', G_binary)

sobel_x = cv.Sobel(img, ddepth=cv.CV_64F, dx=1, dy=0, ksize=3)
sobel_y = cv.Sobel(img, ddepth=cv.CV_64F, dx=0, dy=1, ksize=3)
sobel_tot = np.uint8(np.sqrt(sobel_x ** 2 + sobel_y ** 2))
cv.imshow('sobel', sobel_tot)


cv.waitKey()