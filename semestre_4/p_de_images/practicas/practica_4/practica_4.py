# Transformadas
# Rotar, sesgar, trasladar, Escalar

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("../img/viena.jpeg")
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

cv.imshow("img", img_gray)

# Escalamiento
# pequeña
# imgEsc = cv.resize(img,[250,250])
# cv.imshow("imgEsc1", imgEsc)
# # grande
# imgEsc = cv.resize(img,[800,800])
# cv.imshow("imgEsc2", imgEsc)
# # por coordenadas hacer a la mitad
# imgEsc = cv.resize(img, None, fx=0.5, fy=0.5)
# cv.imshow("imgEsc3", imgEsc)
# # interpolacion lineal
# imgEsc = cv.resize(img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_LINEAR)
# cv.imshow("imgEsc3", imgEsc)

# Escalamiento con matriz afín
# fx=1.5 # 0.5 Escalamiento disparejo
# fy=1.5 # 0.8 Escalamiento disparejo
# rows, cols = np.shape(img_gray)
# M = np.float32([[fx,0,0],[0,fy,0]])
# dst = cv.warpAffine(img_gray,M,[int(cols*fx), int(rows*fy)])
# cv.imshow('rotAfin', dst)

# # Rotacion
# # Solo rota en multiplos de 90
# imgRot90 = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
# cv.imshow('imgRot90', imgRot90)
# # Sentido contrario de los 90
# imgRot90 = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
# cv.imshow('imgRot90Counter', imgRot90)
# # 180 grados
# imgRot180 = cv.rotate(img, cv.ROTATE_180)
# cv.imshow('imgRot180', imgRot180)

# Rotación con matriz afín
# teta esta en radianes habría que convertir = rad(), los grados son sentido horario en negativo y antihorario en positivo

# Rotación desde la esquina superior izquierda
# rows, cols = np.shape(img_gray)

# angle = 45  # Ángulo de rotación en grados
# theta = np.radians(angle)
# cosine = np.cos(theta)
# sine = np.sin(theta)
# M_rotation = np.array([[cosine, -sine, 0],
#                        [sine, cosine, 0]])

# Rotación desde el centro
# angle = 45  # Ángulo de rotación en grados
# theta = np.radians(angle)
# cosine = np.cos(theta)
# sine = np.sin(theta)
# center_x = img_gray.shape[1] // 2
# center_y = img_gray.shape[0] // 2
# M_rotation = np.array([[cosine, -sine, center_x*(1-cosine) + center_y*sine],
#                        [sine, cosine, center_y*(1-cosine) - center_x*sine]])

# rotated_img = cv.warpAffine(img_gray, M_rotation, (cols, rows))

# cv.imshow('Rotacion', rotated_img)
# cv.waitKey(0)
# cv.destroyAllWindows()

# Rotacion  Lilivette
#Rotación con matriz
# grados = 45
# theta = np.radians(grados)
# rows, cols = np.shape(img_gray)
# M= np.float32([[np.cos(theta), np.sin(theta), 0],
#                             [-np.sin(theta), np.cos(theta), 0]]) # Matriz afin
# imgRotAfin = cv.warpAffine(img_gray,M,None)
# cv.imshow('Imagen rotación matriz', imgRotAfin)

# Traslación
# tx = 500  # Traslación en dirección x
# ty = 0  # Traslación en dirección y
# M_translation = np.float32([[1, 0, tx],
#                             [0, 1, ty]])

# translated_img = cv.warpAffine(img_gray, M_translation, [int(cols + tx), int(rows + ty)]) # cuando sumas aquí crece el frame y deja la misma imagen

# cv.imshow('Traslacion', translated_img)

# Tarea Sesgo Vertical y Sesgo Horizontal


cv.waitKey()