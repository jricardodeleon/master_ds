import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("../img/viena.jpeg")

cv.imshow("img", img)


cv.waitKey()