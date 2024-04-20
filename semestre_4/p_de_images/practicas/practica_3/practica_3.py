import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("viena.jpeg")
img2 = cv.imread("eiffel.jpeg")
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

cv.imshow("img", img_gray)

## Operaciones con imagenes
# copia de imagen
# img_copy = img_gray.copy()
# cv.imshow("img2", img_copy)

# # contraste más, más cercano al 255 para eso se multiplica la imagen 'I' por '1' más un factor 'FC'.
# imgHC = img_gray * 1.5
# # Evitar desborde osea que se vean los negros como borrosos o brillosos
# np.putmask(imgHC, imgHC > 255, 255)
# # Se necesita el unit8 para que imshow lo despliegue bien
# imgHC = np.uint8(imgHC)
# cv.imshow("imgHC", imgHC)

# # contraste más, más cercano al 255 para eso se multiplica la imagen 'I' por un factor 'FC' menor que 1.
# imgLC = img_gray * .5
# # Se necesita el unit8 para que imshow lo despliegue bien
# imgLC = np.uint8(imgLC)
# cv.imshow("imgLC", imgLC)

# # Brillo
# # Más brillo
# # contraste más, más cercano al 255 para eso se sumas la imagen 'I' por un factor 'FC' mayor que 1.
# imgHB = np.double(img_gray) + 100
# np.putmask(imgHB, imgHB > 255, 255)
# # Se necesita el unit8 para que imshow lo despliegue bien
# imgHB = np.uint8(imgHB)
# cv.imshow("imgHB", imgHB)                                                     

# # Brillo
# # Menos brillo
# # contraste más, más cercano al 255 para eso se menos la imagen 'I' por un factor 'FC' menor que 1.
# imgLB = np.double(img_gray) - 100
# np.putmask(imgLB, imgLB < 0, 0)
# # Se necesita el unit8 para que imshow lo despliegue bien
# imgLB = np.uint8(imgLB)
# cv.imshow("imgHB", imgLB)

# # complemento
# img_comp = 255 -img_gray
# cv.imshow('img_comp', img_comp)

# Combinación
# print(img_gray.shape)
# print(img2_gray.shape)
# img2 = cv.imread("eiffel.jpeg")
# img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
# img_gray = cv.resize(img_gray,[750,500])
# img_comb = 0.3 * img_gray + 0.7 * img2_gray
# img_comb = np.uint8(img_comb)
# cv.imshow('img_comb', img_comb)


# La siguientes 3 funciona como combinada 
#1
# # img2 = abs(img*alpha + beta)
# img3 = img.copy()
# alpha = 1
# beta = -30
# img3 = cv.convertScaleAbs(img3,alpha=alpha,beta=beta) # altera contraste y brillo
# cv.imshow('img3',img3)

#2
# #dst = src1*alpha + src2*beta + gamma
# beta=0
# gamma = -30
# img4 = cv.addWeighted(img,alpha,img,beta,gamma)
# cv.imshow('img4',img4)

# 3
# alpha = 0.5
# beta = 0.5
# gamma = 0
# img5 = cv.addWeighted(img,alpha,img2,beta,gamma) # altera solo brillo
# cv.imshow('img5',img5)

# funcion para que no se cierren las ventanas
cv.waitKey()