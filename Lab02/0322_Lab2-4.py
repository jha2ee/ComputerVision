#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spyder Editor

Lab 2 - Filter
Gachon University
Computer Vision

@author: jha2ee

# 2-4 Image Pyramid
"""

import cv2 as cv
import numpy as np

# pyramid
src = cv.imread('003.jpeg', cv.IMREAD_COLOR)
height, width, channel = src.shape

dst = cv.pyrDown(src)
dst2 = cv.pyrUp(src, dstsize=(width * 2, height * 2), borderType = cv.BORDER_DEFAULT)
    # 간단히 dst2 = cv2.pyrUp(src) 를 사용해도 됨
    
cv.imshow("src", src)
cv.imshow("pyrDown", dst)
cv.imshow("pyrUp", dst2)

cv.waitKey()
cv.destroyAllWindows()

# blending
rabbit = cv.imread('003.jpeg')
dog = cv.imread('005.jpeg')
print(rabbit.shape)
print(dog.shape)
# [100:300, 150:400] mask 
rabbit_dog = np.hstack((rabbit[:, :256], dog[:, 256:]))

# generate Gaussian pyramid for apple
rabbit_copy = rabbit.copy()
gp_rabbit = [rabbit_copy]
for i in range (6):
    rabbit_copy = cv.pyrDown(rabbit_copy)
    gp_rabbit.append(rabbit_copy)
    
# generate Gaussian pyramid for orange
dog_copy = dog.copy()
gp_dog = [dog_copy]
for i in range (6) :
    dog_copy = cv.pyrDown(dog_copy)
    gp_dog.append(dog_copy)
    
# generate Laplacian Pyramid for apple
rabbit_copy = gp_rabbit[5]
lp_rabbit = [rabbit_copy]
for i in range (5, 0, -1):
    gaussian_expanded = cv.pyrUp(gp_rabbit[i])
    laplacian = cv.subtract(gp_rabbit[i-1], gaussian_expanded)
    lp_rabbit.append(laplacian)
    
# generate Laplacian Pyramid for orange
dog_copy = gp_dog[5]
lp_dog = [dog_copy]
for i in range (5, 0, -1):
    gaussian_expanded = cv.pyrUp(gp_dog[i])
    laplacian = cv.subtract(gp_dog[i-1], gaussian_expanded)
    lp_dog.append(laplacian)
    
# Now add left and right halves of images in each level
pyramid = []
n = 0
for rabbit_lap, dog_lap in zip(lp_rabbit, lp_dog):
    n += 1
    cols, rows, ch = rabbit_lap.shape
    laplacian = np.hstack((rabbit_lap[:, 0:int(cols/2)], dog_lap[:, int(cols/2):]))
    pyramid.append(laplacian)

# now reconstruct
reconstruct = pyramid[0]
for i in range(1, 6):
    reconstruct = cv.pyrUp(reconstruct)
    reconstruct = cv.add(pyramid[i], reconstruct)
        
cv.imshow("rabbit", rabbit)
cv.imshow("dog", dog)
cv.imshow("rabbit_dog", rabbit_dog)
cv.imshow("rabbit_dog_reconstruct", reconstruct)


cv.waitKey(0)
cv.destroyAllWindows()