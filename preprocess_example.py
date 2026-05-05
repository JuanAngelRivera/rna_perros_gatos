import os
import cv2

img = cv2.imread('example/original.jpg')

reescaled = cv2.resize(img, (128, 128))

cv2.imwrite('example/reescaled.jpg', reescaled)

gray = cv2.cvtColor(reescaled, cv2.COLOR_BGR2GRAY)

cv2.imwrite('example/gray.jpg', gray)

_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

cv2.imwrite('example/binary.jpg', binary)

binary_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

cv2.imwrite('example/binary_adaptive.jpg', binary_adaptive)