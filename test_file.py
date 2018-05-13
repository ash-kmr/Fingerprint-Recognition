import cv2 
from zhangsuen2 import ZhangSuen
import numpy as np
from preprocess import preprocess
import features
import helper
from rotation import correctrotation
import constants as const

def get_most_similar(fv1,fv2):

	s = 0
	x,y = None,None
	for i in range(len(fv1)):
		for j in range(len(fv2)):
			si = helper.similarity(fv1[i],fv2[j])

			if si>s:
				x=i
				y=j 
				s = si

	return x,y,s

def match_level(pv1,pv2, fv1, fv2):

	ml = np.zeros((len(pv1),len(pv2)))

	for i in range((len(pv1))):
		for j in range((len(pv2))):
			if np.all(np.abs(pv1[i]-pv2[j]) > const.BG):
				continue

			ml[i,j] = 0.5 + 0.5*helper.similarity(fv1[i], fv2[j])
			


	ml_prime = np.zeros((len(pv1),len(pv2)))

	for i,row in enumerate(ml):
		j = np.argmax(row)
		ml_prime[i,j] = row[j]

	ml = ml_prime
	ml_prime = np.zeros((len(pv1),len(pv2)))

	for j,col in enumerate(ml.T):
		i = np.argmax(col)
		ml_prime[i,j] = col[i]

	# print(ml_prime)

	print(ml_prime.sum()/max(len(pv1),len(pv2)))






class MyTest:

	def __init__(self, img1, img2):

		self.image = img1

		#img2 = 255 - img2
		rows, cols = img2.shape
		#M = cv2.getRotationMatrix2D((cols/2,rows/2),37,1)
		#dst = cv2.warpAffine(img2,M,(cols,rows))
		#dst = 255 - dst
		self.rotated = img2
		#self.rotated = dst

	def original_stuff(self):

		img2 = self.image
		angle,xc,yc = correctrotation(img2)

		img2 = 255 - img2
		rows, cols = img2.shape
		M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
		dst = cv2.warpAffine(img2,M,(cols,rows))
		dst = 255 - dst
		self.image = dst

		cv2.imwrite("1.jpg", self.image)

		image, m, orientations = preprocess(self.image)
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
		fincoords = z.remove_minutiae(coords, cv2.imread("1.jpg", 0))
		# rotatecoords, angle, maskedimage = z.rotate_minutiae(fincoords, cv2.imread("1.jpg", 0))
		# cv2.imwrite("minutiaeextracted.jpg", (maskedimage)*255)
		vector = z.get_ridge_count(fincoords, image)
		feature_vectors = features.get_features(fincoords,vector,orientations)

		return feature_vectors

	def rotated_stuff(self):

		cv2.imwrite("rot.jpg", self.rotated)
		img2 = cv2.imread("rot.jpg", 0)
		angle,xc,yc = correctrotation(img2)

		if angle<0:
			angle = -angle

		img2 = 255 - img2
		rows, cols = img2.shape
		M = cv2.getRotationMatrix2D((cols/2,rows/2),-angle,1)
		dst = cv2.warpAffine(img2,M,(cols,rows))
		dst = 255 - dst
		self.rotated = dst


		cv2.imwrite("2.jpg", self.rotated)
		image, m, orientations = preprocess(self.rotated)
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
		cv2.imwrite("thinnedimage2.jpg", (1-img)*255)
		print "dome"
		coords, mask = z.extractminutiae(img)
		cv2.imwrite("minu2.jpg", mask*255 )
		fincoords = z.remove_minutiae(coords, cv2.imread("2.jpg", 0))
		# rotatecoords, angle, maskedimage = z.rotate_minutiae(fincoords, cv2.imread("2.jpg", 0))
		# cv2.imwrite("minutiaeextracted2.jpg", (maskedimage)*255)
		vector = z.get_ridge_count(fincoords, image)
		feature_vectors = features.get_features(fincoords,vector,orientations)

		return feature_vectors


	def convert_to_polar(self, obj, index):

		base = obj[index]

		polars = []
		polar_obj = []
		for current in obj:
			cur = features.FeaturePolar(current,base)
			polars.append(cur.convert())
			polar_obj.append(cur)

		return polars, polar_obj




