#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spyder Editor

Lab 2 - Filter
Gachon University
Computer Vision

@author: jha2ee

# 2-1 Simple Image Processing
"""

import cv2 as cv 
import sys
from matplotlib import pyplot as plt

img = cv.imread('003.jpeg')
              
if img is None:
    sys.exit(" 파일을 찾을 수 없습니다.")
    
print('Image Size = ', img.shape) #image size [height width channel]
h, w, c = img.shape

'''
    
OpenCV - -> BGR image
'''
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

roi = img[100:300, 150:400] # image cropping (ROI extraction by slicing)

cv.imshow('Color Image', img) 
cv.imshow('Gray Image', gray)
cv.imshow('Cropped Image', roi)
"""
matplotlib --> RGB image
"""
RGB_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
RGB_roi = RGB_img[100:300, 150:400]
roi = img[100:300, 150:400] # image cropping (ROI extraction by slicing)

plt.imshow(RGB_roi)
plt.show()

cv.waitKey()
cv.destroyAllWindows()