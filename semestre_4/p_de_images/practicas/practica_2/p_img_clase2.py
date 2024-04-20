import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("viena.jpeg")

cv.imshow("img", img)

# Método del promedio
b,g,r = cv.split(img)

# imgProm = np.uint8((np.double(b) + np.double(g) + np.double(r)) / 3)
# cv.imshow('imgPromedio', imgProm)

# # Metodo Luma - BT.601
# imgLuma601 = np.uint8(((np.double(b) * .114) + (np.double(g) * .587) + (np.double(r) * .299)))
# cv.imshow('imgLuma601', imgLuma601)

# # Metodo Luma - BT.709
# imgLuma709 = np.uint8(((np.double(b) * .0722) + (np.double(g) * .7152) + (np.double(r) * .2126)))
# cv.imshow('imgLuma709', imgLuma709)

# # Metodo Luma - BT.2020
# imgLuma2020 = np.uint8(((np.double(b) * .0593) + (np.double(g) * .6780) + (np.double(r) * .2627)))
# cv.imshow('imgLuma2020', imgLuma2020)

# Método Saturación
# imgSat = np.uint8(((np.maximum(np.double(b), np.double(g), np.double(r)) + np.minimum(np.double(b), np.double(g), np.double(r)))) / 2)
# cv.imshow('imgSat', imgSat)

# Descomposición Mínima
# imgDesMin = np.uint8(np.minimum(np.double(b), np.double(g), np.double(r)))
# cv.imshow('imgDesMin', imgDesMin)

# Descomposición Máxima
imgDesMin = np.uint8(np.minimum(np.double(b), np.double(g), np.double(r)))
cv.imshow('imgDesMin', imgDesMin)

# Con OpenCV
imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # Equivalente a BT:601
cv.imshow('imgGray', imgGray)

# funcion para que no se cierren las ventanas
cv.waitKey()
