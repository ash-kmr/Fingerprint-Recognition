import numpy as np

def dki(p1,p2):
	return np.sqrt(np.sum(p1-p2)**2)

def dfi(t1,t2):

	diff = t1-t2

	if diff>(-np.pi) and diff <= np.pi:
		return diff

	elif (diff<=(-np.pi)):
		return 2*np.pi + diff

	elif (diff>np.pi):
		reurn 2*np.pi - diff
