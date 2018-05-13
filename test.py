import cv2 
from zhangsuen2 import ZhangSuen
import numpy as np
from preprocess import preprocess
import features
# image = cv2.imread("enh.jpg", 0)

image = cv2.imread("102_2.jpg",0)
image, m, orientations = preprocess(image)

for i in range(image.shape[0]):
	for j in range(image.shape[1]):
		if image[i][j] > 50: image[i][j] = 1
		else: image[i][j] = 0
print("done")
cop = image.copy()
#cv2.imwrite("intermediate.jpg", image*255)
z = ZhangSuen(image)
img = z.performThinning()
thinned = img.copy()
cv2.imwrite("thinnedimage.jpg", (1-img)*255)
print "dome"
coords, mask = z.extractminutiae(img)
cv2.imwrite("minu.jpg", mask*255 )
fincoords = z.remove_minutiae(coords, cv2.imread("102_2.jpg", 0))
rotatecoords, angle, maskedimage = z.rotate_minutiae(fincoords, cv2.imread("102_2.jpg", 0))
cv2.imwrite("minutiaeextracted.jpg", (maskedimage)*255)
vector = z.get_ridge_count(fincoords, image)

feature_vectors = features.get_features(fincoords,vector,orientations)
print(feature_vectors)
