import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

imagen = cv.imread("../img/viena.jpeg")

cv.imshow("img orig", imagen)

# Obtener las dimensiones de la imagen
alto, ancho, _ = imagen.shape
print("imagen original", imagen.shape)

# Definir los factores de escala
fx = .5
fy = .5

# Definir la matriz de transformación
T = np.array([[fx, 0, 0],
              [0, fy, 0],
              [0, 0, 1]])

# Calcular la matriz inversa de T
T_inv = np.linalg.inv(T)

# Inicializar la imagen escalada
imagen_escala = np.zeros((int(alto*fy), int(ancho*fx), 3), dtype=np.uint8)

# Aplicar la transformación
for y in range(int(alto*fy)):
    for x in range(int(ancho*fx)):
        # Aplicar la inversa de la transformación a las coordenadas de salida
        coordenadas_salida = np.array([x, y, 1])  # Coordenadas homogéneas
        coordenadas_originales = T_inv @ coordenadas_salida
        
        # Deshacer la normalización homogénea
        x_orig = int(coordenadas_originales[0] / coordenadas_originales[2])
        y_orig = int(coordenadas_originales[1] / coordenadas_originales[2])
        
        # Verificar si las coordenadas están dentro de los límites de la imagen original
        if 0 <= x_orig < ancho and 0 <= y_orig < alto:
            # Copiar el píxel de la imagen original a la imagen escalada
            imagen_escala[y][x] = imagen[y_orig][x_orig]

print("imagen final", imagen_escala.shape)

# Guardar la imagen escalada con un nuevo nombre
cv.imwrite('imagen_escalada.png', imagen_escala)

# Mostrar la imagen escalada en pantalla
cv.imshow('Imagen escalada', imagen_escala)

print("Imagen escalada guardada como 'imagen_escalada.png'")

cv.waitKey()
