import cv2 
from zhangsuen2 import ZhangSuen
import numpy as np
from preprocess import preprocess
import features
import helper
import os
import sys

from test_file import MyTest, get_most_similar, match_level
# image = cv2.imread("enh.jpg", 0)

# dir = r'DB1_B'
# images = [file for file in os.listdir(dir)]
# print images

# for i in images:
	# for j in images:

# image = cv2.imread("my_db/6_2.jpg",0)
# img2 = cv2.imread("my_db/6_1.jpg",0)
image = cv2.imread(sys.argv[1],0)
img2 = cv2.imread(sys.argv[2],0)

# image = cv2.imread("my_db/6_4.jpg",0)
# img2 = cv2.imread("my_db/6_1.jpg",0)


# image, m, orientations = preprocess(image)

# for i in range(image.shape[0]):
# 	for j in range(image.shape[1]):
# 		if image[i][j] > 50: image[i][j] = 1
# 		else: image[i][j] = 0
# print("done")
# cop = image.copy()
# #cv2.imwrite("intermediate.jpg", image*255)
# z = ZhangSuen(image)
# img = z.performThinning()
# thinned = img.copy()
# cv2.imwrite("thinnedimage.jpg", (1-img)*255)
# print "dome"
# coords, mask = z.extractminutiae(img)
# cv2.imwrite("minu.jpg", mask*255 )
# fincoords = z.remove_minutiae(coords, cv2.imread("102_2.jpg", 0))
# rotatecoords, angle, maskedimage = z.rotate_minutiae(fincoords, cv2.imread("102_2.jpg", 0))
# cv2.imwrite("minutiaeextracted.jpg", (maskedimage)*255)
# vector = z.get_ridge_count(fincoords, image)
# vecto2 = z.get_ridge_count(rotatecoords, image)


# feature_vectors = features.get_features(fincoords,vector,orientations)
# feature_vectors_2 = features.get_features(rotatecoords,vecto2,orientations)
# # print(feature_vectors)

test1 = MyTest(image, img2)

fv1,fo1 = test1.original_stuff()
fv2,fo2 = test1.rotated_stuff()

# print("A",fv1)
# print("B",fv2)
# print(helper.similarity(fv1[0], fv2[0]))

sl = get_most_similar(fv1,fv2)

b1 = sl[0]
b2 = sl[1]

pv1, po1 = test1.convert_to_polar(fo1,b1)
pv2, po2 = test1.convert_to_polar(fo2,b2)

ml = match_level(pv1,pv2,fv1,fv2)
print(ml, "Matched" if (ml>0.3) else "Not Matched")


