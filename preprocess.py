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
		A = self.A.copy()
		flat = A.copy().flatten()
		hist = np.histogram(flat, bins = range(256))
		hist_percentile = np.percentile(hist, q)
		hist_50 = np.percentile(hist, 50)

		classified = self.classified.copy()

		# 1 for ridge, 2 for valley
		ridge = (self.A <= hist_percentile)
		valley = (self.A >= hist_50)
		valley = valley*2

		classified = ridge + valley

		for i in range(A.shape[0]):
			for j in range(A.shape[1]):
				if classified[i,j] is 0:
					T = classified[(i-2):(i+2), (j-2):(j+2)]
					flat_T = T.copy().flatten()
					hist_T = np.histogram(flat_T, bins = range(256))
					hist_30 = np.percentile(hist_T, 30)

					if A[i,j] >= hist_30:
						# valley
						classified[i,j] = 2
					else:
						classified[i,j] = 1

		# 1 matlab ridge and 0 matlab valley
		self.classified = classified == 1  



