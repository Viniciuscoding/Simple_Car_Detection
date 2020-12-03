#!/usr/bin/env python
# coding: utf-8


#import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)
matplotlib.rcParams['image.cmap'] = 'gray'


image = cv2.imread('quiz1_car.png')


imageCopy = image.copy()
#plt.imshow(image[:,:,::-1]);
#plt.title("Original Image")

imageB, imageG, imageR = np.moveaxis(image, 2, 0)

plt.figure(figsize=(20,12))
plt.subplot(141)
plt.imshow(image[:,:,::-1]);
plt.title("Original Image")
plt.subplot(142)
plt.imshow(imageB);
plt.title("Blue Channel")
plt.subplot(143)
plt.imshow(imageG);
plt.title("Green Channel")
plt.subplot(144)
plt.imshow(imageR);
plt.title("Red Channel");


thresh = 115
maxValue = 255
src = imageB

th, dst_bin = cv2.threshold(src, thresh, maxValue, cv2.THRESH_BINARY)


plt.imshow(dst_bin, cmap='gray', vmin=0, vmax=255)
plt.title("Threshold Binary")



kSize5 = (5,5)
kSize11 = (11,11)
kSize23 = (23,23)
kSize45 = (45,45)
kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kSize5)
kernel11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kSize11)
kernel23 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kSize23)
kernel45 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kSize45)


imageEroded1 = cv2.erode(dst_bin, kernel23, iterations=3)
imageDilated1 = cv2.dilate(imageEroded1, kernel23, iterations=3)



plt.figure(figsize=[20,12])
plt.subplot(231);plt.imshow(imageEroded1, cmap='gray', vmin=0, vmax=255);plt.title("Threshold Binary Eroded 3 Times");
plt.subplot(232);plt.imshow(imageDilated1, cmap='gray', vmin=0, vmax=255);plt.title("Threshold Binary Dilated 3 Times");


contours, hierarchy = cv2.findContours(imageDilated1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


for i in range(len(contours)):
    area = cv2.contourArea(contours[i])
    perimeter = cv2.arcLength(contours[i],True)
 
    print("Contour {} has area = {} and perimeter = {}".format(i, area, perimeter))


image_c = image.copy()
cv2.drawContours(image_c, contours, 0, (0,255,0), 5)
plt.subplot(233);plt.imshow(image_c[:,:,::-1], cmap='gray', vmin=0, vmax=255);plt.title("Car Identified");
plt.title("Image With All Contours Detected")
