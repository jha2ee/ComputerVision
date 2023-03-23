# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Exercise 1 imread and imshow
'''
import cv2 as cv
import sys

img = cv.imread('bus.jpg')

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')
    
cv.imshow('Image Display', img)

cv.waitKey()
cv.destroyAllWindows()
'''

# Exercise 2 imwrite
'''
import cv2 as cv
import sys

img = cv.imread('bus.jpg')

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')
    
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

cv.imwrite('bus_gray.jpg', gray)
cv.imshow('Color Image', img)
cv.imshow('Gray Image', gray)

cv.waitKey()
cv.destroyAllWindows()
'''

# Exercise 3-1
'''
import cv2 as cv
import sys

img = cv.imread('bus.jpg')

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')
    
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

(thresh, binary_img) = cv.threshold(gray, 120, 255, cv.THRESH_BINARY)

cv.imwrite('bus_bin.jpg', binary_img)

cv.imshow('Color Image', img)
cv.imshow('Gray Image', gray)
cv.imshow('Binary Image', binary_img)

cv.waitKey()
cv.destroyAllWindows()
'''

# Exercise 3-2
'''
import cv2 as cv
import sys

img = cv.imread('bus.jpg')

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')
    
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

(thresh, binary_otsu_img) = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

print('The Best Threshold Value obtained by Otsu = ', thresh)

cv.imwrite('bus_bin_otsu.jpg', binary_otsu_img)

cv.imshow('Gray Image', gray)
cv.imshow('Binary Otsu Image', binary_otsu_img)

cv.waitKey()
cv.destroyAllWindows()
'''

# Exercise 3-3
'''
import cv2 as cv
import sys
import matplotlib.pyplot as plt

img = cv.imread('bus.jpg')

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')
    
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

histogram = cv.calcHist([gray], [0], None, [256], [0, 256])
plt.plot(histogram, color='b', linewidth = 3)
'''

# Exercise 4-1
'''
import cv2 as cv
import sys
import matplotlib.pyplot as plt

img = cv.imread('bus.jpg')

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')
    

# Order -> BGR

# red
image_red = img.copy()
image_red[:, :, 1] = 0 # Delete Green
image_red[:, :, 0] = 0 # Delete Blue

# green
image_green = img.copy()
image_green[:, :, 2] = 0 # Delete Red
image_green[:, :, 0] = 0 # Delete Blue

# blue
image_blue = img.copy()
image_blue[:, :, 2] = 0 # Delete Red
image_blue[:, :, 1] = 0 # Delete Green

# Display
cv.imshow('Color Image', img)

cv.imshow('Red Channel', image_red)
cv.imshow('Green Channel', image_green)
cv.imshow('Blue Channel', image_blue)

cv.waitKey()
cv.destroyAllWindows()
'''

# Exercise 4-2
'''
import cv2 as cv
import sys
import matplotlib.pyplot as plt

img = cv.imread('bus.jpg')

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')
    
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

hist1 = cv.calcHist([img], [0], None, [256], [0, 256]) #B
hist2 = cv.calcHist([img], [1], None, [256], [0, 256]) #G
hist3 = cv.calcHist([img], [2], None, [256], [0, 256]) #R

plt.subplot(221), plt.plot(hist1, color='b')
plt.subplot(222), plt.plot(hist2, color='g')
plt.subplot(223), plt.plot(hist3, color='r')

plt.xlim([0, 256])

plt.show()
'''

# Exercise 4-3
import cv2 as cv
import sys
import matplotlib.pyplot as plt

img = cv.imread('bus.jpg')

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')
    
colors = ['b', 'g', 'r']
bgr_planes = cv.split(img)

for (p, c) in zip(bgr_planes, colors):
    histogram = cv.calcHist([p], [0], None, [256], [0, 256])
    plt.plot(histogram, color=c, linewidth = 3)

plt.show()