import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

imagen = cv.imread("../img/viena.jpeg")

cv.imshow("img", imagen)

# Obtener las dimensiones de la imagen
alto, ancho, _ = imagen.shape

print(imagen.shape)

# Definir los factores de escala
fx = 2
fy = 2

# Definir la matriz de transformación
T = np.array([[fx, 0, 0],
              [0, fy, 0],
              [0, 0, 1]])

# Inicializar la imagen escalada
imagen_escala = np.zeros((int(alto*fy), int(ancho*fx), 3), dtype=np.uint8)

# Aplicar la transformación
for y in range(int(alto*fy)):
    for x in range(int(ancho*fx)):
        # Aplicar la inversa de la transformación a las coordenadas de salida
        x_orig = int(x / fx)
        y_orig = int(y / fy)
        
        # Verificar si las coordenadas están dentro de los límites de la imagen original
        if x_orig < ancho and y_orig < alto:
            # Copiar el píxel de la imagen original a la imagen escalada
            imagen_escala[y][x] = imagen[y_orig][x_orig]
            
print(imagen_escala.shape)

# Guardar la imagen escalada con un nuevo nombre
cv.imwrite('imagen_escalada.png', imagen_escala)

# Mostrar la imagen escalada en pantalla
cv.imshow('Imagen escalada', imagen_escala)

cv.waitKey()