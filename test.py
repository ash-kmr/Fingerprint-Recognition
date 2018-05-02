import cv2 
from zhangsuen import ZhangSuen
import numpy as np
from preprocess import Preprocess
image = cv2.imread("cap.jpg", 0)
for i in range(image.shape[0]):
	for j in range(image.shape[1]):
		if image[i][j] > 50: image[i][j] = 0
		else: image[i][j] = 1
print("done")
z = ZhangSuen(image)
img = z.performThinning()
img = Preprocess(image).extractminutiae(img)
cv2.imwrite("minutiaeextracted.jpg", img)