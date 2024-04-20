import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("../img/viena.jpeg")

# cv.imshow("img", img)

# imgB = img.copy()
# imgB[:,:,1] = 0
# imgB[:,:,2] = 0

# imgG = img.copy()
# imgG[:,:,0] = 0
# imgG[:,:,2] = 0

# imgR = img.copy()
# imgR[:,:,1] = 0
# imgR[:,:,0] = 0


# h = cv.calcHist([imgB],[0],None,[256],[0,255])
# plt.plot(h,color='blue')
# h = cv.calcHist([imgG],[1],None,[256],[0,255])
# plt.plot(h,color='green')
# h = cv.calcHist([imgR],[2],None,[256],[0,255])
# plt.plot(h,color='red')
# plt.xlabel('Niveles de intensidad')
# plt.ylabel('Frecuencia')
# plt.show()


# En escala de grises

img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
h = cv.calcHist([img], None, None, [256], [0,255])
plt.plot(h, color="gray")
H = h.cumsum()
h = plt.hist(img.flatten(), 256)
H = plt.hist(img.flatten(), bins=256, cumulative=True )
plt.show()