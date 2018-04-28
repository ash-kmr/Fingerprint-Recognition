import sys
import numpy as np
import cv2
from zhangsuen import ZhangSuen
import math
import matplotlib.pyplot as plt


from utils import estimateOrientations, showOrientations, showImage

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

	def showImage(self, image, label, vmin=0.0, vmax=1.0):
		plt.figure().suptitle(label)
		plt.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
		# plt.show()

	def showOrientations(self, image, orientations, label, w=32, vmin=0.0, vmax=1.0):
		self.showImage(image, label)
		height, width = image.shape
		for y in range(0, height, w):
			for x in range(0, width, w):
				if np.any(orientations[y:y+w, x:x+w] == -1.0): continue

				cy = (y + min(y + w, height)) // 2
				cx = (x + min(x + w, width)) // 2

				orientation = orientations[y+w//2, x+w//2]

				plt.plot(
						[cx - w * 0.5 * np.cos(orientation),
							cx + w * 0.5 * np.cos(orientation)],
						[cy - w * 0.5 * np.sin(orientation),
							cy + w * 0.5 * np.sin(orientation)],
						'r-', lw=1.0)

		plt.show()

	def averageOrientation(self, orientations):
		"""
		Calculate the average orientation in an orientation field.
		"""

		orientations = np.array(orientations).flatten()
		o = orientations[0]

		# If the difference in orientation is more than 90 degrees, accoridngly change the orientation
		aligned = np.where(np.absolute(orientations - o) > np.pi/2,
				np.where(orientations > o, orientations - np.pi, orientations + np.pi),
				orientations)

		return np.average(aligned) % np.pi, np.std(aligned)
		

	def getOrientations(self, image, w=16):

		height, width = image.shape

		# Apply Guassian Filter to smooth the image
		image = cv2.GaussianBlur(image,(3,3),0)

		# Compute the gradients gx and gy at each pixel
		gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
		gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

		# Estimate the local orientation of each block
		xblocks, yblocks = height // w, width // w
		orien = np.empty((xblocks, yblocks))
		for i in range(xblocks):
			for j in range(yblocks):
				denominator, numerator = 0, 0
				for v in range(w):
					for u in range(w):
						numerator += 2 * gx[i*w+v, j*w+u] * gy[i*w+v, j*w+u]
						denominator += gx[i*w+v, j*w+u] ** 2 - gy[i*w+v, j*w+u] ** 2

				orien[i, j] = np.arctan2(numerator, denominator)/2

		# Rotate the orientations by 90 degrees
		orien = (orien + np.pi/2) % np.pi

		# Smooth the orientation field
		orientations = np.full(image.shape, -1.0)
		o = np.empty(orien.shape)

		# pad it with 0 since 5 by 5 filter
		orien = np.pad(orien, 2, mode="edge")

		for y in range(xblocks):
			for x in range(yblocks):
				surrounding = orien[y:y+5, x:x+5]
				orientation, deviation = self.averageOrientation(surrounding)
				if deviation > 0.5:
					orientation = orien[y+2, x+2]
				o[y, x] = orientation
		orien = o

		orientations = np.full(image.shape, -1.0)
	
		for y in range(xblocks):
			for x in range(yblocks):
				orientations[y*w:(y+1)*w, x*w:(x+1)*w] = orien[y, x]

		return orientations


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
# o = (pre.orientationField(16))
o =  pre.getOrientations(pre.B)
print(o)
# showImage(pre.B, "j")
pre.showOrientations(pre.B, o, "i", 16)

# cv2.imwrite('image.jpg' ,pre.classified)

# zh = ZhangSuen(pre.classified)
# cv2.imwrite('image-thin.jpg', zh.performThinning()*255)

# except:
	# print("Error")
