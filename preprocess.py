import sys
import numpy as np
import cv2
from zhangsuen import ZhangSuen
import math

class Preprocess:
	""" Class for preprocessing fingerprint for recognition """
	def __init__(self, fingerprint, alpha = 150, gamma = 95):
		self.A = fingerprint
		self.mean = fingerprint.mean()
		self.stddev = fingerprint.std()
		self.alpha = alpha
		self.gamma = gamma
		self.classified = np.zeros(fingerprint.shape)
		self.B = np.zeros(fingerprint.shape)
		assert self.gamma > self.stddev , "gamma cannot be smaller than mean of image array"

	def stretchDistribution(self):
		""" function for streching the distribution of image
		for image enhancement """
		self.B = self.alpha + self.gamma*((self.A - self.mean)/self.stddev)

	def orientationField(self, w):
		img = self.B.copy()

		

		r,c = img.shape

		blocks = []

		for i in range(0, r, w):
			for j in range(0,c,w):
				block = img[i:(i+w), j:(j+w)]
				blocks.append(block)

		angles = []
		for block in blocks:
			gx = cv2.Sobel(block,cv2.CV_64F,1,0,ksize=3)
			gy = cv2.Sobel(block,cv2.CV_64F,0,1,ksize=3)

			# print(gx)

			numerator = 0
			denominator = 0
			for i in range(w):
				for j in range(w):
					numerator = numerator + 2*gx[i,j]*gy[i,j]
					denominator = denominator + (gx[i,j]*gx[i,j] - gy[i,j]*gy[i,j])



			angle = (math.atan(numerator/denominator))/2

			if math.isnan(angle):
				continue
			else:
				angles.append(angle)

		print(angles)




	def binarization(self, q):
		B = self.B.copy()
		B = B/255
		flat = B.copy().flatten()
		hist1 = np.histogram(flat, density=True)
		hist = hist1[0]
		hist_percentile = np.percentile(hist, q)
		hist_50 = np.percentile(hist, 50)

		classified = self.classified.copy()

		print(hist_percentile)

		# 1 for ridge, 2 for valley
		ridge = (self.B <= hist_percentile).astype(int)
		# print(hist)
		valley = (self.B >= hist_50).astype(int)
		# print(hist_50)
		valley = valley*2

		classified = ridge + valley

		for i in range(B.shape[0]):
			for j in range(B.shape[1]):
				if classified[i,j] is 0:
					T = classified[(i-2):(i+2), (j-2):(j+2)]
					flat_T = T.copy().flatten()
					flat_T = flat_T/(np.amax(flat_T))
					hist_T1 = np.histogram(flat_T, density=True)
					hist_T = hist_T1[0]
					hist_30 = np.percentile(hist_T, 30)

					if B[i,j] >= hist_30:
						# valley
						classified[i,j] = 2
					else:
						classified[i,j] = 1

		# 1 matlab ridge and 0 matlab valley
		self.classified = (classified == 1 ).astype(int) 
		self.classified = self.classified

# try:
file_name = sys.argv[1]
img = cv2.imread(file_name, 0)
pre = Preprocess(img)
pre.stretchDistribution()
# pre.binarization(25)
pre.orientationField(16)
# cv2.imwrite('image.jpg' ,pre.classified)

# zh = ZhangSuen(pre.classified)
# cv2.imwrite('image-thin.jpg', zh.performThinning()*255)

# except:
	# print("Error")
