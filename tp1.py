from __future__ import print_function
from __future__ import division
import cv2 
import numpy as np
import argparse
from matplotlib import pyplot as plt


#Convert them to HSV format:
hsv_base = cv2.imread('im1.jpg')
hsv_test1 = cv2.imread('im2.jpg')
hsv_test2 = cv2.imread('im3.jpg')

#Check up 
if hsv_base is None or hsv_test1 is None or hsv_test2 is None:
    print('Could not open or find the images!')
    exit(0)

 

# Initialize the arguments to calculate the histograms (bins, ranges and channels H and S ):

histSize = [256]

# hue varies from 0 to 179, saturation from 0 to 255

s_ranges = [0,255]
ranges = s_ranges # concat lists
# Use the 0-th and 1-st channels
channels = [0]

#Calculate the Histograms for the base image, the 2 test images and the half-down base image:
hist_base = cv2.calcHist([hsv_base], channels, None, histSize, ranges)
cv2.normalize(hist_base, hist_base, 0, 1, cv2.NORM_MINMAX)


hist_test1 = cv2.calcHist([hsv_test1], channels, None, histSize, ranges)
cv2.normalize(hist_test1, hist_test1, 0, 1, cv2.NORM_MINMAX)

hist_test2 = cv2.calcHist([hsv_test2], channels, None, histSize, ranges)
cv2.normalize(hist_test2, hist_test2, 0, 1, cv2.NORM_MINMAX)

# Apply sequentially the 4 comparison methods between the histogram of the base image (hist_base) and the other histograms:

#Apply 1st method: HISTCMP_CORREL
base_base = cv2.compareHist(hist_base, hist_base, cv2.HISTCMP_CORREL)
base_test1 = cv2.compareHist(hist_base, hist_test1, cv2.HISTCMP_CORREL)
base_test2 = cv2.compareHist(hist_base, hist_test2,cv2.HISTCMP_CORREL)

print('---------------Method: Correlation, Image originale :',base_base, '/ im1 : ', base_test1, '/ im2 : ', base_test2,'---------------')
if (base_test1 > base_test2) :
   close = "im1" 
else :
   close = "im2"
print("using Correlation method",close, ' is the closest one to the original image ')
#Apply 2nd method : Chi-Square  CV_COMP_CHISQR

""" base_base = cv2.compareHist(hist_base, hist_base, cv2.HISTCMP_CHISQR)
base_test1 = cv2.compareHist(hist_base, hist_test1, cv2.HISTCMP_CHISQR)
base_test2 = cv2.compareHist(hist_base, hist_test2,cv2.HISTCMP_CHISQR)

print('Method: chi-square, Image originale :',base_base, '/ im1 : ', base_test1, '/ im2 : ', base_test2)

 """
#Applying a 3rd method : Dinter
base_base = cv2.compareHist(hist_base, hist_base, cv2.HISTCMP_INTERSECT)
base_test1 = cv2.compareHist(hist_base, hist_test1, cv2.HISTCMP_INTERSECT)/base_base
base_test2 = cv2.compareHist(hist_base, hist_test2,cv2.HISTCMP_INTERSECT)/base_base

print('---------------Method: Intersection, Image originale :',base_base/base_base, '/ im1 : ', base_test1, '/ im2 : ', base_test2,'---------------')

if (base_test1 > base_test2) :
   close = "im1" 
else :
   close = "im2"
print("using Intersection method",close, ' is the closest one to the original image ')

plt.plot(hist_base,'-b', label='Original Image')
plt.plot(hist_test1,'--r', label='image2')
plt.plot(hist_test2,'y',label='image3')

plt.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)
cv2.imshow('image originale',hsv_base)
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()