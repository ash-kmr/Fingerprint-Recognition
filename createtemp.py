import cv2
import numpy as np

image = cv2.imread("first/original.jpg", 0)
image = 255-image
rows, cols = image.shape
M = cv2.getRotationMatrix2D((cols/2,rows/2),-10,1)
dst = cv2.warpAffine(image,M,(cols,rows))
dst = 255-dst
cv2.imwrite("first/template2.jpg", dst)