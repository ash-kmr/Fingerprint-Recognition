import numpy as np
import weave
import cv2
import copy
from helper import segment
from rotation import correctrotation, rotate
from sklearn.neighbors import NearestNeighbors
import numpy as np
from lineimitator import createLineIterator

class ZhangSuen:
	def __init__(self, image):
		self.image = image
		self.r, self.c = image.shape

	def step(self, iter):
		I = self.image
		M = np.ones(self.image.shape)
		a, b = self.image.shape
		thin = """
		for (int i = 1; i < a; i++){
			for (int j = 0; j < b; j++){
				int p2 = I2(i-1, j);
				int p3 = I2(i-1, j+1);
				int p4 = I2(i, j+1);
				int p5 = I2(i+1, j+1);
				int p6 = I2(i+1, j);
				int p7 = I2(i+1, j-1);
				int p8 = I2(i, j-1);
				int p9 = I2(i-1, j-1);
				int k = 0;
				int A = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
                    (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
                    (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                    (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
                int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
                int m1 = iter == 1 ? (p2 * p4 * p6) : (p2 * p4 * p8);
				int m2 = iter == 1 ? (p4 * p6 * p8) : (p2 * p6 * p8);
				if (A == 1 && B >= 2 && B <= 6 && m1 == 0 && m2 == 0) {
                	M2(i,j) = 0;
            	}
			}
		}
		"""
		weave.inline(thin, ["I", "iter", "M", "a", "b"])
		return np.multiply(I,M)
	def performThinning(self):
		#print(self.image.shape)
		while True:
			prev = self.image.copy()
			self.image = self.step(1)
			self.image = self.step(2)
			diff = np.absolute(self.image - prev).sum()
			if diff == 0:
				break
		#print(self.image.shape)
		return self.image

	def extractminutiae(self, image):
		coords = []
		mask = np.zeros(image.shape)
		for i in range(1, image.shape[0]-1):
			for j in range(1, image.shape[1]-1):
				p = [image[i,j+1],image[i-1,j+1],image[i-1,j],image[i-1,j-1],image[i,j-1],image[i+1,j-1],image[i+1,j],image[i+1,j+1],image[i,j+1]]
				CN = 0
				for k in range(len(p)-1):
					CN += abs(p[k] - p[k+1])
				CN = CN/2
				#print(CN)
				if image[i][j] == 1:
					if CN == 1:
						coords.append((i, j, 1))
						mask[i, j] = 1
					if CN == 3:
						coords.append((i, j, 3))
						mask[i, j] = 1
		return coords, mask

	def remove_minutiae(self, coords, image):
		#print(image.shape)
		image, segmentfilter = segment(image)
		mask = np.zeros(image.shape)
		finalcoords = []
		for i, j, t in coords:
			if i > 6 and i < image.shape[0] - 7 and j > 6 and j < image.shape[1] - 7:
				block = segmentfilter[i-5:i+6, j-5:j+6]
				#print(block.sum())
				if block.sum() == 121:
					#mask[i, j] = 1
					finalcoords.append((i, j, t))

		fincoords = []
		for i, j, t in finalcoords:
			count = 0
			for x, y, t2 in finalcoords:
				if x != i or y != j:
					if x < i+6 and x > i-6:
						if y < j+6 and y > j-6:
							count = 1
							break
			if count == 1: continue
			fincoords.append((i, j, t))
			mask[i, j] = 1

		cv2.imwrite("minwithmask.jpg", mask*255)
		return fincoords

	def rotate_minutiae(self, coords, image):
		mask = np.zeros(image.shape)
		rows, cols = image.shape
		angle, r, c = correctrotation(image)
		angle = angle*np.pi/180
		rotatedcoords = []
		for x, y, t in coords:
			xd, yd = rotate([r/2, c/2], [x, y], angle)
			rotatedcoords.append((xd, yd, t))
		#print rotatedcoords
		for x, y, t in rotatedcoords:
			mask[int(round(x)), int(round(y))] = 1
		return rotatedcoords, angle, mask

	def myfunction(self, rcoords, image):
		coords = np.array([[i, j] for i, j, k in rcoords])
		nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(coords)
		distances, indices = nbrs.kneighbors(coords)
		ridgecount = []
		for i in range(len(coords)):
			p1, p2 = np.array(coords[indices[i][1]]), np.array(coords[indices[i][2]])
			r = np.array(coords[i])
			iter1 = createLineIterator(r, p1, image)
			print(iter1)
			break

