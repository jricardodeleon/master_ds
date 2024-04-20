import cv2 as cv
import numpy as np

img = cv.imread("../img/mario.jpeg", cv.IMREAD_GRAYSCALE)
cv.imshow("img original", img)

# Obtener las dimensiones de la imagen
alto, ancho = img.shape
theta = np.radians(45)

fx = 2
fy = 2

# Crear la matriz de transformación combinada para escalamiento y rotación
# Nota: Esta matriz asume que la rotación se realiza primero y luego el escalamiento
F = np.array([[fx * np.cos(theta), -fy * np.sin(theta), 0],
              [fx * np.sin(theta), fy * np.cos(theta), 0],
              [0, 0, 1]])

# Calcular las nuevas coordenadas extremas de la imagen después de la transformación
coordenadas_extremas = np.array([[0, 0, 1],
                                 [alto - 1, 0, 1],
                                 [0, ancho - 1, 1],
                                 [alto - 1, ancho - 1, 1]])
coordenadas_transformadas = coordenadas_extremas @ F
minx, miny = np.min(coordenadas_transformadas[:, :2], axis=0)
maxx, maxy = np.max(coordenadas_transformadas[:, :2], axis=0)

# Calcular las nuevas dimensiones de la imagen
alto_nuevo = int(np.ceil(maxx - minx))
ancho_nuevo = int(np.ceil(maxy - miny))

# Crear una imagen de salida con las nuevas dimensiones
imgEsc = np.zeros((alto_nuevo, ancho_nuevo), dtype=np.uint8)

# Aplicar la transformación
for x in range(alto):
    for y in range(ancho):
        # Aplicar la transformación
        v = np.array([x, y, 1]) @ F
        # Ajustar las coordenadas para el centro de la imagen de salida
        x_new = int(np.round(v[0] - minx))
        y_new = int(np.round(v[1] - miny))
        # Verificar si las nuevas coordenadas están dentro de los límites de la imagen de salida
        if 0 <= x_new < alto_nuevo and 0 <= y_new < ancho_nuevo:
            imgEsc[x_new, y_new] = img[x, y]

# Mostrar la imagen escalada y rotada
            
cv.imwrite('imagen_rotada.jpeg', imgEsc)
cv.imshow('Imagen escalada y rotada', imgEsc)

cv.waitKey()
