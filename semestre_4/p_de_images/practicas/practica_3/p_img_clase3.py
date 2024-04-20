import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("viena.jpeg")
img2 = cv.imread("eiffel.jpeg")

cv.imshow("img", img)

# # contraste más, más cercano al 255 para eso se multiplica la imagen 'I' por '1' más un factor 'FC'.
# imgHC = img * 1.5
# # Evitar desborde osea que se vean los negros como borrosos o brillosos
# np.putmask(imgHC, imgHC > 255, 255)
# # Se necesita el unit8 para que imshow lo despliegue bien
# imgHC = np.uint8(imgHC)
# cv.imshow("imgHC", imgHC)


# # contraste más, más cercano al 255 para eso se multiplica la imagen 'I' por un factor 'FC' menor que 1.
# imgLC = img * .5
# # Se necesita el unit8 para que imshow lo despliegue bien
# imgLC = np.uint8(imgLC)
# cv.imshow("imgLC", imgLC)

# # Brillo
# # Más brillo
# # contraste más, más cercano al 255 para eso se sumas la imagen 'I' por un factor 'FC' mayor que 1.
# imgHB = np.double(img) + 100
# np.putmask(imgHB, imgHB > 255, 255)
# # Se necesita el unit8 para que imshow lo despliegue bien
# imgHB = np.uint8(imgHB)
# cv.imshow("imgHB", imgHB)  

# # Brillo
# # Menos brillo
# # contraste más, más cercano al 255 para eso se menos la imagen 'I' por un factor 'FC' menor que 1.
# imgLB = np.double(img) - 100
# np.putmask(imgLB, imgLB < 0, 0)
# # Se necesita el unit8 para que imshow lo despliegue bien
# imgLB = np.uint8(imgLB)
# cv.imshow("imgHB", imgLB)

# # complemento
# img_comp = 255 - img
# cv.imshow('img_comp', img_comp)

# Combinación
print(img.shape)
print(img2.shape)
img2 = cv.imread("eiffel.jpeg")
img = cv.resize(img,[750,500])
img_comb = 0.3 * img + 0.3 * img2
img_comb = np.uint8(img_comb)
cv.imshow('img_comb', img_comb)


cv.waitKey()