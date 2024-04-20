import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("../img/lennacolor.png")
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

cv.imshow('img_gray', img_gray)

# Mejora de Nitidez con laplaciano, mediante resaltar los bordes de la imagen
kernel_laplaciano = np.array([ [0,1,0], [1, -4,1], [0,1,0]])
Laplace = cv.filter2D(np.double(img_gray), -1, kernel_laplaciano)
img_nitida = img_gray - Laplace

cv.imshow('img_nitida', cv.convertScaleAbs(img_nitida))


cv.waitKey()