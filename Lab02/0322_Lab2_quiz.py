#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spyder Editor

Lab 2 - Filter
Gachon University
Computer Vision

@author: jha2ee

# 2-2 Image Smoothing
"""

import cv2 as cv 
import sys
from matplotlib import pyplot as plt
import numpy as np

#img = cv.imread('noise_quiz1.png')
#img = cv.imread('noise_quiz2.png')
img = cv.imread('noise_quiz3.png')
              
if img is None:
    sys.exit("파일을 찾을 수 없습니다.")
    
    
# Image Blurring (Image Smoothing) using 2D convolution (image filtering)

# 1. Averaging
"""
OpenCV --> BGR image
"""

kernel = np.ones((5, 5), np.float32)/25 # kernal setting
# Using 1/9 average filter
dst = cv.filter2D(img, -1, kernel) # output image

cv.imshow('Original Image', img)
cv.imshow('Averaging Image', dst)

# 2. Gaussian Blurring (Image Smoothing) using 2D convolution (image filtering)
# img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
 
dst = cv.GaussianBlur(img, (5, 5), 1) # output image (Sigma = 3)
dst2 = cv.GaussianBlur(img, (5, 5), 3) # output image (Sigma = 10)
dst3 = cv.GaussianBlur(img, (0, 0), 7) # output image (Sigma = 5)


cv.imshow('Original Image', img)
cv.imshow('Gaussian Image sigma = 1 with 5x5', dst)
cv.imshow('Gaussian Image sigma = 3 with 5x5', dst2)
cv.imshow('Gaussian Image sigma = 7', dst3)

cv.waitKey()
cv.destroyAllWindows()

# 3. median Blurring
dst = cv.medianBlur(img, 1)
dst2 = cv.medianBlur(img, 3)
dst3 = cv.medianBlur(img, 5)

cv.imshow('Original Image', img)
cv.imshow('MedianBlur Image sigma = 1 with 5x5', dst)
cv.imshow('MedianBlur Image sigma = 3 with 5x5', dst2)
cv.imshow('MedianBlur Image sigma = 5', dst3)

cv.waitKey()
cv.destroyAllWindows()
