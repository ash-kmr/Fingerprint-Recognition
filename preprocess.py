import sys
import numpy as np
import cv2
from zhangsuen import ZhangSuen
import math
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import scipy.signal as signal
import gabor
import helper


from utils import *

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
		self.B = ((self.A - self.mean)/self.stddev)

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


	def rotatedRectWithMaxArea(self,image, angle):
		# https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders/16778797#16778797

		h, w = image.shape

		width_is_longer = w >= h
		side_long, side_short = (w, h) if width_is_longer else (h, w)

		# since the solutions for angle, -angle and 180-angle are all the same,
		# if suffices to look at the first quadrant and the absolute values of sin,cos:
		sin_a, cos_a = abs(np.sin(angle)), abs(np.cos(angle))
		if side_short <= 2.0 * sin_a * cos_a * side_long:
			# half constrained case: two crop corners touch the longer side,
			# the other two corners are on the mid-line parallel to the longer line
			x = 0.5 * side_short
			wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
		else:
			# fully constrained case: crop touches all 4 sides
			cos_2a = cos_a * cos_a - sin_a * sin_a
			wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a

		image = ndimage.interpolation.rotate(image, np.degrees(angle), reshape=False)

		hr, wr = int(hr), int(wr)
		y, x = (h - hr) // 2, (w - wr) // 2

		return image[y:y+hr, x:x+wr]

	def averagePeak(self, peaks, x_signature):

		pre = peaks[0]
		avg = 0
		for i in range(1,len(peaks)):
			avg = avg + (peaks[i] - pre)
			pre = peaks[i]

		return avg/(len(peaks)-1)



	def getFrequencies(self, image, orientations, w=16):

		height, width = image.shape

		xblocks, yblocks = height//w, width//w

		F = np.empty((xblocks, yblocks))

		for x in range(xblocks):
			for y in range(yblocks):
				orientation_window = orientations[(x*w+w)//2, (y*w+w)//2]
				block = image[y*w:(y+1)*w, x*w:(x+1)*w]
				# Rotate the block so its normal to the ridge
				block = self.rotatedRectWithMaxArea(block, np.pi/2 + orientation_window) 

				if block.size == 0:
					F[x,y] = -1
					continue

				x_signature = np.zeros(w)

				for k in range(w):
					for d in range(w):
						u = x*w + (d-w/2)*np.cos(orientation_window) + (k-w/2)*np.sin(orientation_window)
						v = y*w + (d-w/2)*np.sin(orientation_window) + (w/2-k)*np.cos(orientation_window)
						
						x_signature[k] = x_signature[k] + image[int(u),int(v)]

					x_signature[k] = x_signature[k]/w

				
				peaks = signal.find_peaks_cwt(x_signature, np.array([3]))
				if len(peaks) < 2:
					F[x, y] = -1
				else:
					f = self.averagePeak(peaks, x_signature)
					F[x, y] = 1 / f


		# frequencies = cv2.GaussianBlur(F,(3,3),0)
		frequencies = F
		return frequencies


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

	def applyGabor(self, image, orientations, frequencies ,w=16):

		height, width = image.shape
		xblocks, yblocks = height//w, width//w

		for x in range(xblocks):
			for y in range(yblocks):
				angle = orientations[(x*w+w)//2, (y*w+w)//2]
				lamda = 1/(frequencies[x, y])
				stddev = np.std(image[x*w:(x+1)*w, y*w:(y+1)*w])
				print(angle, lamda)
				kernel = cv2.getGaborKernel((w,w), stddev , angle, lamda, 0, 0, ktype=cv2.CV_32F)
				image[x*w:(x+1)*w, y*w:(y+1)*w] = cv2.filter2D(image[x*w:(x+1)*w, y*w:(y+1)*w],cv2.CV_8UC1, kernel)

		return image


# try:


file_name = sys.argv[1]
img = cv2.imread(file_name, 0)
pre = Preprocess(img)
pre.stretchDistribution()
# pre.binarization(25)
# o = (pre.orientationField(16))
print("Filtering")
image = pre.B
print("Normalizing")
image = normalize(image)

print("Finding mask")
mask = findMask(image)

print("Applying local normalization")
image = np.where(mask == 1.0, localNormalize(image), image)
 
o = np.where(mask== 1.0, helper.getOrientations(image), -1.0 ) 
showOrientations(image, o, 'i')
f = np.where(mask == 1.0, helper.getFrequencies(image, o), -1.0)

image = gabor.gaborFilter(image, o, f)
image = np.where(mask == 1.0, image, 1.0)
image = np.where(mask == 1.0, binarize(image), 1.0)
showImage(image, "binarized")
plt.show()

# zh = ZhangSuen(pre.classified)
# cv2.imwrite('image-thin.jpg', zh.performThinning()*255)

# except:
	# print("Error")
