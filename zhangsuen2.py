import numpy as np
import weave

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
		while True:
			prev = self.image.copy()
			self.image = self.step(1)
			self.image = self.step(2)
			diff = np.absolute(self.image - prev).sum()
			if diff == 0:
				break

		return self.image*255
	
	def extractminutiae(self, image):
		mask = np.zeros(image.shape)
		for i in range(1, image.shape[0]-1):
			for j in range(1, image.shape[1]-1):
				p = [image[i-1, j-1], image[i-1, j], image[i, j-1], image[i+1, j], image[i+1, j+1], image[i, j+1], image[i-1, j+1], image[i+1, j-1]]
				p = np.array(p)
				#if(p.sum()>0) : print(p)
				if image[i][j] == 1:
					if p.sum() == 1 or p.sum() > 2:
						print(p)
						mask[i][j] = 1
		return mask*255
