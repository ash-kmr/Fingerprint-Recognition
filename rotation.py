from sklearn.cluster import KMeans
import cv2
import math
import numpy as np

def correctrotation(image):
	"""
	find the amount to rotation required to correct the orientation of a
	fingerprint image.
	"""
	points = [[i, j, image[i, j]] for i in range(image.shape[0]) for j in range(image.shape[1]) if image[i][j] < 60]
	points = np.array(points)

	kmeans = KMeans(n_clusters = 2, random_state = 3).fit(points)
	mat = kmeans.cluster_centers_
	num = abs(mat[0][0] - mat[1][0])
	den = abs(mat[0][1] - mat[1][1])
	angle = np.arctan(num/den)*180/np.pi
	print mat
	print num
	print den
	if(mat[0][1] < mat[1][1]): return angle-90, (mat[0][0]+mat[1][0])/2, (mat[0][1]+mat[1][1])/2
	else: return 90-angle, (mat[0][0]+mat[1][0])/2, (mat[0][1]+mat[1][1])/2


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy



# test
img = cv2.imread("102_2.jpg", 0)
img = 255 - img
rows, cols = img.shape
M = cv2.getRotationMatrix2D((cols/2,rows/2),0,1)
dst = cv2.warpAffine(img,M,(cols,rows))
dst = 255 - dst
cv2.imwrite("rotated.jpg", dst)
angle = correctrotation(dst)
print angle
