import numpy as np

class ZhangSuen:
	def __init__(self, image):
		self.image = image
		self.r, self.c = image.shape

	def A(i, j):
		canvas = self.image[(i-1)%self.r:(i+2)%self.r, (j-1)%self.c:(j+2)%self.c].copy()
		canvas = canvas.flatten().tolist()
		canvas = canvas[1:4] + canvas[5:] + canvas[:2]
		n_transitions = 0
		prev = canvas[0]
		for i in range(1, 9):
			if canvas[i] == 1 and prev == 0:
				n_transitions += 1
			prev = canvas[i]
		cond1 = canvas[0]*canvas[2]*canvas[4]
		cond2 = canvas[2]*canvas[4]*canvas[6]
		cond3 = canvas[0]*canvas[2]*canvas[6]
		cond4 = canvas[0]*canvas[4]*canvas[6]
		return n_transitions, cond1, cond2

	def B(i, j):
		canvas = self.image[(i-1)%self.r:(i+2)%self.r, (j-1)%self.c:(j+2)%self.c].copy()
		return canvas.sum()

	def step(self, iter):
		mask = np.ones(self.r, self.c)
		changed = 0
		for i in range(self.r):
			for j in range(self.c):
				n_transitions, cond1, cond2, cond3, cond4 = A(i, j)
				if self.image[i, j] != 1 and B(i, j) > 1:
					continue
				if B(i, j) < 2 or B(i, j) > 6:
					continue
				if n_transitions != 1:
					continue
				if cond1 != 0 and iter == 1:
					continue
				if cond2 != 0 and iter == 1:
					continue
				if cond3 != 0 and iter == 2:
					continue
				if cond4 != 0 and iter == 2:
					continue
				mask[i, j] = 0
				changed = 1
		self.image = self.image*mask
		return changed

	def performThinning(self):
		while True:
			step1 = step(1)
			step2 = step(2)
			if step1 == 0 and step2 == 0:
				break

		return self.image.copy()