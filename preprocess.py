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

