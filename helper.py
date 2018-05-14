import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
import scipy.signal as signal
import cv2
import constants as const


def standard_normalization(image):
	""" 
	Returns an image with 0 mean and 1 STD
	"""
	image = np.copy(image)
	mean = np.mean(image)
	std = np.std(image)

	image = (image-mean)/std
	return image

def segment(im,w=16,thresh=0.1):
	
	rows,cols = im.shape;    
	im = standard_normalization(im);    # normalise to get zero mean and unit standard deviation
	
	# print(im.shape)
	new_rows, new_cols = int(w*np.ceil(rows*1.0/w)), int(w*np.ceil(cols*1.0/w)) 
	xblocks, yblocks = new_rows//w, new_cols//w
	
	padded_img = np.zeros((w*xblocks,w*yblocks));
	stddevim = np.zeros((new_rows,new_cols));
	# print(cols)
	padded_img = im;
	
	for x in range(xblocks):
		for y in range(yblocks):
			block = padded_img[x*w:(x+1)*w, y*w:(y+1)*w];
			stddevim[x*w:(x+1)*w, y*w:(y+1)*w] = np.std(block)
	
	stddevim = stddevim[0:rows, 0:cols]
	
	# print(im.shape)
	# print(stddevim.shape)
	mask = stddevim > thresh
	
	mean_val = np.mean(im[mask]);
	
	std_val = np.std(im[mask]);
	
	normim = (im - mean_val)/(std_val);
	
	return(normim,mask)

def normalize(image):
	"""
	Normalizes the given image
	"""
	image = np.copy(image)
	image -= np.min(image)
	m = np.max(image)
	if m > 0.0:
		# Image is not just all zeros
		image *= 1.0 / m
	return image

def localNormalize(image, w=16):
	"""
	Normalizes each block
	"""
	image = np.copy(image)
	height, width = image.shape
	for y in range(0, height, w):
		for x in range(0, width, w):
			image[y:y+w, x:x+w] = normalize(image[y:y+w, x:x+w])

	return image

def custom_normalization(image, mo = 100, varo = 100):
	"""
	Returns an image with a custom mean mo and varience varo
	From: Fingerprint image enhancement: Algorithm and performance evaluation, 1998
	"""

	image = np.copy(image)
	mean = np.mean(image)
	std = np.std(image)

	image = np.where(image > mean, mo + np.sqrt((varo*(image-mean)**2)/(std**2)), mo - np.sqrt((varo*(image-mean)**2)/(std**2)))
	return image


def stretchDistribution(image, alpha = 150, gamma = 95):
	"""
	Stretches the distrubition for an enhanced image
	From: Implementation of An Automatic Fingerprint Identification System, IEEE EIT 2007
	"""
	image = alpha + gamma*(standard_normalization(image))
	return image


def binarize(image, w=16):

	image = np.copy(image)
	height, width = image.shape
	for y in range(0, height, w):
		for x in range(0, width, w):
			block = image[y:y+w, x:x+w]
			threshold = np.average(block)
			image[y:y+w, x:x+w] = np.where(block >= threshold, 1.0, 0.0)

	return image


def getOrientations(image, w=16):
	"""
	Get the Orientation Map

	Based on Fingerprint image enhancement: Algorithm and performance evaluation, 1998

	:params
	:image, the input image
	:w, block size

	"""

	height, width = image.shape

	# Apply Guassian Filter to smooth the image
	image = cv2.GaussianBlur(image,(5,5),0)

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
	orientation = np.empty(orien.shape)

	# pad it with 0 since 3 by 3 filter, t gave better result than 5x5
	orien = np.pad(orien, 1, mode="edge")

	for x in range(xblocks):
		for y in range(yblocks):
			surrounding = orien[x:x+3, y:y+3]
			cos_angles = np.cos(2*surrounding)
			sin_angles = np.sin(2*surrounding)
			cos_angles = np.mean(cos_angles)
			sin_angles = np.mean(sin_angles)
			orientation[x,y] = np.arctan2(sin_angles,cos_angles)/2

	for x in range(xblocks):
		for y in range(yblocks):
			orientations[x*w:(x+1)*w, y*w:(y+1)*w] = orientation[x, y]

	return orientations

def rotatedRectWithMaxArea(image, angle):

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


def block_freq(block, angle):
	"""
	Frequency of a block
	"""
	proj = np.sum(block, axis=0)
	proj = normalize(proj)
	
	peaks = signal.find_peaks_cwt(proj, np.array([3]))
	freq = -1
	if len(peaks) > 1:
		f = (peaks[-1] - peaks[0])/(len(peaks)-1)
		if f>=5 and f<=15:
			freq = 1.0/f

	return freq

	

def getFrequencies(image, orientations, w=16):
	"""
	Get the Freuencies

	Based on Fingerprint image enhancement: Algorithm and performance evaluation, 1998

	:params
	:image, the input image
	:orientation, the orientation map
	:w, block size

	"""
	height, width = image.shape
	xblocks, yblocks = height // w, width // w
	F = np.empty((xblocks, yblocks))
	for x in range(xblocks):
		for y in range(yblocks):
			orientation = orientations[x*w+w//2, y*w+w//2]
			block = image[x*w:(x+1)*w, y*w:(y+1)*w]
			block = rotatedRectWithMaxArea(block, np.pi/2 + orientation)
			F[x,y] = block_freq(block, orientation)

	frequencies = np.full(image.shape, -1.0)
	F = np.pad(F, 1, mode="edge")
	for x in range(xblocks):
		for y in range(yblocks):
			surrounding = F[x:x+3, y:y+3]
			surrounding = surrounding[np.where(surrounding > 0.0)]  
			if surrounding.size == 0:
				frequencies[x*w:(x+1)*w, y*w:(y+1)*w] = 1
			else:   
				frequencies[x*w:(x+1)*w, y*w:(y+1)*w] = np.median(surrounding)
			
	return frequencies

def binarize(image, w=16):
	
	image = np.copy(image)
	height, width = image.shape
	for y in range(0, height, w):
		for x in range(0, width, w):
			block = image[y:y+w, x:x+w]
			threshold = np.average(block)
			image[y:y+w, x:x+w] = np.where(block >= threshold, 1.0, 0.0)

	return image


def dki(p1,p2):
	""" Returns the euclidian distance """
	return np.sqrt(np.sum(p1-p2)**2)

def dfi(t1,t2):

	diff = t1-t2

	if diff>(-np.pi) and diff <= np.pi:
		return diff

	elif (diff<=(-np.pi)):
		return 2*np.pi + diff

	elif (diff>np.pi):
		return 2*np.pi - diff


def similarity(Fl,Ft):

	W = const.W
	Fl = np.array(Fl)
	Ft = np.array(Ft)

	diff = np.abs(Fl-Ft)

	dot_product = W.dot(diff.T)
	# print(const.bl)

	if dot_product > const.bl:
		return 0 

	else:
		return ((const.bl - dot_product)/const.bl)



def shiftcorrection(image):
    """
    function for correcting the placement of a fingerprint in an image.

    Takes an image as input and returns the image in which fingerprint 
    is shifted to the center.
    """
    img = image.copy()
    xmax, xmin, ymax, ymin = -1, 10000, -1, 10000
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] < 100:
                if i > xmax : xmax = i
                if i < xmin : xmin = i
                if j > ymax : ymax = j
                if j < ymin : ymin = j

    rows,cols = img.shape
    xtrans = (img.shape[0]-xmax-xmin)/2
    ytrans = (img.shape[1]-ymax-ymin)/2
    print("translation")
    print(xtrans, ytrans)
    cv2.imwrite("pretrans.jpg", img)
    M = np.float32([[1,0,ytrans],[0,1,xtrans]])
    dst = cv2.warpAffine(255-img.copy(),M,(cols,rows))
    dst = 255-dst
    cv2.imwrite("posttrans.jpg", dst)
    #cv2.imwrite("shifted.jpg", dst)
    return dst

def cropfingerprint(image):
    img = image.copy()
    xmax, xmin, ymax, ymin = -1, 10000, -1, 10000
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if img[i, j] == 1:
                if i > xmax : xmax = i
                if i < xmin : xmin = i 
                if j > ymax : ymax = j
                if j < ymin : ymin = j

    return img[xmin:xmax+1, ymin:ymax+1], xmax, xmin, ymax, ymin




