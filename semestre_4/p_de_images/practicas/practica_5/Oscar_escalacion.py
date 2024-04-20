import cv2 as cv
import numpy as np

img = cv.imread("../img/viena.jpeg")
imgGray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("img original", img)

alto, ancho, _ = img.shape

fx = 2
fy = 2

# Crear la matriz de transformación usando los factores de escala
F = np.array([[fx, 0, 0],
              [0, fy, 0],
              [0, 0, 1]])
print (F)
# Inicializar la imagen escalada para una imagen en escala de grises (1 canal)
imgEsc = np.zeros((int(alto * fy), int(ancho * fx)), dtype=np.uint8)

print(imgEsc.shape)
# Verificar las coordenadas resultantes antes de acceder a la imagen escalada
for x in range(imgGray.shape[0]):
    for y in range(imgGray.shape[1]):
        # Aplicar la inversa de la transformación a las coordenadas de salida
        v = np.array([x, y, 1]) @ F
        v_rounded = np.round(v).astype(int)  # Redondear las coordenadas y convertirlas a enteros
        # Verificar si las coordenadas están dentro de los límites de la imagen escalada
        if 0 <= v_rounded[0] < imgEsc.shape[0] and 0 <= v_rounded[1] < imgEsc.shape[1]:
            imgEsc[v_rounded[0], v_rounded[1]] = imgGray[x, y]


# # Mostrar la imagen escalada
cv.imshow('Imagen escalada', imgEsc)
cv.waitKey(0)