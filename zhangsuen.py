import numpy as np
# from numba import jit, jitclass


class ZhangSuen:

	def __init__(self, image):
		self.image = image
		self.r, self.c = image.shape
	

	def step(self, iteration):
		image, mask = self.image, np.ones(self.image.shape, np.uint8)
		a, b = mask.shape
		counter = 0
		for i in range(a-1):
			for j in range(b-1):
				p2, p3, p4, p5, p6, p7, p8, p9 = image[i-1, j], image[i-1, j+1], image[i, j+1], image[i+1, j+1], image[i+1, j], image[i+1, j-1], image[i, j-1], image[i-1, j-1]
				A = 0
				l = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
				for k in range(8):
					if l[k] == 0 and l[k+1] == 1:
						A += 1
				B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
				m1 , m2 = 0, 0
				if iteration == 1:
					m1 = p2*p4*p6
					m2 = p4*p6*p8
				elif iteration == 2:
					m1 = p2*p4*p8
					m2 = p2*p6*p8
				if A == 1 and image[i, j] == 1 and B >= 2 and B <= 6 and m1 == 0 and m2 == 0:
					mask[i, j] = 0
					counter += 1
		return np.multiply(self.image, mask), counter


	def performThinning(self):
		while True:
			self.image, c1 = self.step(1)
			self.image, c2 = self.step(2)
			print(c1+c2)
			if c1+c2 == 0:
				break

		return self.image