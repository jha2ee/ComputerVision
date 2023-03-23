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

#img = cv.imread('Lena_color.png')
#gray = cv.imread('Lena_gray.png')
img = cv.imread('003.jpeg')
gray = cv.imread('003_gray.jpeg')
    
              
if img is None:
    sys.exit("파일을 찾을 수 없습니다.")
    
    
# Image Blurring (Image Smoothing) using 2D convolution (image filtering)

# 1. Averaging
"""
OpenCV --> BGR image
"""

kernel = np.ones((3, 3), np.float32)/9 # kernal setting
# Using 1/9 average filter
dst = cv.filter2D(img, -1, kernel) # output image
dst_gray = cv.filter2D(gray, -1, kernel) # output image

cv.imshow('Original Image', img)
cv.imshow('Averaging Image', dst)
cv.imshow('Original Image Gray', gray)
cv.imshow('Averaging Image Gray', dst_gray)

"""
matplotlib --> RGB image
"""

RGB_img = cv.cvtColor(img, cv.COLOR_BGR2RGB) # convert color space from BGR to RGB

kernel = np.ones((5, 5), np.float32)/25 # kernal setting
dst2 = cv.filter2D(RGB_img, -1, kernel) # output image (opencv is applied)
dst2_gray = cv.filter2D(gray, -1, kernel) # output image

plt.subplot(121), plt.imshow(RGB_img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(dst2), plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()


plt.subplot(121), plt.imshow(gray), plt.title('Original Gray')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(dst2), plt.title('Averaging Gray')
plt.xticks([]), plt.yticks([])
plt.show()

cv.waitKey()
cv.destroyAllWindows()


# 2. Gaussian Blurring (Image Smoothing) using 2D convolution (image filtering)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
 
dst = cv.GaussianBlur(gray, (5, 5), 1) # output image (Sigma = 3)
dst2 = cv.GaussianBlur(gray, (5, 5), 3) # output image (Sigma = 10)
dst3 = cv.GaussianBlur(gray, (0, 0), 7) # output image (Sigma = 5)


cv.imshow('Original Image', gray)
cv.imshow('Gaussian Image sigma = 1 with 5x5', dst)
cv.imshow('Gaussian Image sigma = 3 with 5x5', dst2)
cv.imshow('Gaussian Image sigma = 7', dst3)

cv.waitKey()
cv.destroyAllWindows()

"""
matplotlib --> RGB image
"""

gray = cv.imread('003.jpeg')

dst_gray = cv.GaussianBlur(gray, (5, 5), 1) # output image (Sigma = 3)

plt.subplot(121), plt.imshow(gray), plt.title('Original Gray')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(dst_gray), plt.title('Gaussian Gray')
plt.xticks([]), plt.yticks([])
plt.show()

