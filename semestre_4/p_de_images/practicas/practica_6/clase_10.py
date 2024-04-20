import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("../img/edificio.jpg")
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

img = cv.resize(img, dsize=None, fx=0.3, fy=0.3)
cv.imshow('img', img)

imgGauss = cv.GaussianBlur(np.double(img), ksize=(3,3), sigmaX=0)
cv.imshow('imgGauss', np.uint8(imgGauss))

Lx = np.array([ [0,1,0], [0, -2,0], [0,1,0]])
Laplace_x = cv.filter2D(np.double(imgGauss), -1, Lx)
min = np.min (Laplace_x)
imgLaplace_x = Laplace_x - min
max = np.max (imgLaplace_x)
imgLaplace_x = np.uint8((imgLaplace_x / max) * 255)

cv.imshow ('Laplace_x', Laplace_x)
cv.imshow('imgLaplace_x',np.uint8(imgLaplace_x))

Ly = np.array([ [0,0,0], [1, -2,1], [0,0,0]])
Laplace_y = cv.filter2D(np.double(imgGauss), -1, Ly)
min = np.min(Laplace_y)
imgLaplace_y = Laplace_y - min
max = np.max(imgLaplace_y)
imgLaplace_y = np.uint8((imgLaplace_y / max) * 255)
cv.imshow ('imgLaplace_y', imgLaplace_y)

imgLaplaceTotal = Laplace_x + Laplace_y
cv. imshow ('imgLaplaceTotal',np.uint8 (imgLaplaceTotal))

L = np.array([ [0,1,0], [1, -4,1], [0,1,0]])
Laplace_l = cv.filter2D(np.double(imgGauss), -1, L)
cv.imshow('imgLaplace_l', np.uint8(Laplace_l))

imgLaplacian = cv.Laplacian(imgGauss, ddepth=cv.CV_64F)
cv. imshow ('imgLaplacian' ,np.uint8 (imgLaplacian))

cv.waitKey()
