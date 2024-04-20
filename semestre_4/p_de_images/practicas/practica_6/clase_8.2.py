import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa

img = cv.imread("../img/mario.jpeg")
imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('img', img)

aug = iaa.SaltAndPepper(p=0.05)
imgNoise = aug.augment_image(imgGray)
cv.imshow('Image Noise', imgNoise)

imgMedian=cv.medianBlur(imgNoise,3)
cv.imshow('Imagen Median',imgMedian)

from scipy import ndimage
imgMinimum = ndimage.minimum_filter(imgNoise,3)
cv.imshow('imgMinimum',imgMinimum)

imgMaximum = imgMaximum = ndimage.maximum_filter(imgNoise,3)
cv.imshow('imgMaximum',imgMaximum)

cv.waitKey()