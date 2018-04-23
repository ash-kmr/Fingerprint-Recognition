import sys
import numpy as np
import cv2

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
		self.classified = (classified == 2 ).astype(int) 
		self.classified = self.classified*255

# try:
file_name = sys.argv[1]
img = cv2.imread(file_name, 0)
pre = Preprocess(img)
pre.stretchDistribution()
pre.binarization(30)
cv2.imwrite('image.jpg' ,pre.B)

# except:
	# print("Error")
