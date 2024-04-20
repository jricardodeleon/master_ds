import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("../img/mario.jpeg")
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('img', img)

### Funciones que hacen Blur, mismo resultado
imgBlur = cv.blur(img, (9,9))
cv.imshow('imgBlur', imgBlur)

imgBlur2 = cv.boxFilter(img, -1, (9,9))
cv.imshow('imgBlur2', imgBlur2)

kernel = np.ones((9,9), np.float32 )/81
imgBlur3 = cv.filter2D(img,-1,kernel)
cv.imshow('imgBlur3', imgBlur3)

imgGauss = cv.GaussianBlur(img,[9,9],0 ) # tiene un blur mas suavisado
cv.imshow('imgGauss', imgGauss)

kernelGauss = np.float32([[0,1,2,1,0], [1,3,5,3,1], [2,5,9,5,2], [1,3,5,3,1], [0,1,2,1,0]]) / 57 # tiene un blur mas suavisado
imgGauss2 = cv.filter2D(img,-1,kernelGauss)
cv.imshow('imgGauss2', imgGauss2)




cv.waitKey()