# Detección de esquinas
# Traquin de objetos o deteccíon de alguna figura dentro de una imagen. Elementos prominentes en una imagen.
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("../img/figuras.png")
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

img_gray = cv.resize(img_gray, dsize=None, fx=0.4, fy=0.4)
cv.imshow('img_gray', img_gray)

imgGrayBlur = cv.blur (np.double(img_gray),ksize=(3,3))
cv. imshow ('imgGrayBlur' ,np.uint8 (imgGrayBlur))

Ix = cv.Sobel(imgGrayBlur, ddepth=cv.CV_64F, dx=1, dy=0, ksize=3)
Iy = cv.Sobel(imgGrayBlur, ddepth=cv.CV_64F, dx=0, dy=1, ksize=3)

# Calcular la matrix de correlación
A = Ix ** 2
B = Iy ** 2
C = Ix * Iy

# Aplicar filtro Gausean a las matrices A, B, C
A = cv.GaussianBlur(A, ksize=(5,5), sigmaX=0)
B = cv.GaussianBlur(B, ksize=(5,5), sigmaX=0)
C = cv.GaussianBlur(C, ksize=(5,5), sigmaX=0)

# experimentamos con un valor de alpha (a) de .1
# calculamos las esquinas exponenciales

a = 0.1
V = ((A * B) - C ** 2) - a * ((A + B) ** 2)

# umbral th
th = 1000
U = (V > th).astype(int)

# visualizamos como vamos 
cv.imshow('U', np.uint8(U *  255))




cv.waitKey()