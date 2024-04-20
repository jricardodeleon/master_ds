import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("../img/caballo.jpg")
cv.imshow("img", img)

# # Histograma acumulado original
# img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# h = cv.calcHist([img], None, None, [256], [0,255])
# plt.plot(h, color="gray")
# H = h.cumsum()
# h = plt.hist(img.flatten(), 256)
# H = plt.hist(img.flatten(), bins=256, cumulative=True )
# plt.show()

# # Histograma ecualizado
# ieq = cv.equalizeHist(img)
# ieq = cv.cvtColor(ieq, cv.COLOR_BGR2GRAY)
# h = cv.calcHist([ieq], None, None, [256], [0,255])
# plt.plot(h, color="gray")
# H = h.cumsum()
# h = plt.hist(img.flatten(), 256)
# H = plt.hist(img.flatten(), bins=256, cumulative=True )
# plt.show()

# leq = cv.equalizeHist(img)
# H = plt.hist(leq.flatten(), bins = 256, cumulative = True, range = [0,255])

# cv.imshow('equalizada', np.uint8(leq))

# hist = cv.calcHist([img], None, None,[256],[0,256])
# plt.plot(hist, color = "gray")
# plt.show()

cv.waitKey()