#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spyder Editor

Lab 2 - Filter
Gachon University
Computer Vision

@author: jha2ee

# 2-3 Image Gradients
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

#img = cv.imread('003_gray.jpeg', 0)
img = cv.imread('Lena_gray.png')
#img = cv.imread('edge1.png')
#img = cv.imread('edge2.png')
#img = cv.imread('edge3.png')


# Sobel
sobelx = cv.Sobel(img, -1, 1, 0, 3)
sobely = cv.Sobel(img, -1, 0, 1, 3)

abs_grad_x = cv.convertScaleAbs(sobelx)
abs_grad_y = cv.convertScaleAbs(sobely)
sobel = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

mask = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), plt.imshow(sobel, cmap='gray')
plt.title('Sobel'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3), plt.imshow(sobelx, cmap='gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4), plt.imshow(sobely, cmap='gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

cv.imshow('Original', img)
cv.imshow('Sobel X', sobelx)
cv.imshow('Sobel Y', sobely)
cv.imshow('Sobel', sobel)

# Canny
threshold1 = 130
threshold2 = 180
canny_img = cv.Canny(img, threshold1, threshold2)

cv.imshow('Original', img)
cv.imshow('Canny', canny_img)

# Laplacian

laplacian = cv.Laplacian(img, cv.CV_8U, ksize = 3)

mask1 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
mask2 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
mask3 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

laplacian1 = cv.filter2D(img, -1, mask1)
laplacian2 = cv.filter2D(img, -1, mask2)
laplacian3 = cv.filter2D(img, -1, mask3)
laplacian4 = cv.Laplacian(img, -1)

cv.imshow('Original', img)
cv.imshow('Laplacian 1', laplacian1)
cv.imshow('Laplacian 2', laplacian2)
cv.imshow('Laplacian 3', laplacian3)
cv.imshow('Laplacian 4', laplacian4)

cv.waitKey()
cv.destroyAllWindows()

#