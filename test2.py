import cv2 
from zhangsuen2 import ZhangSuen
import numpy as np
image = cv2.imread("enh.jpg", 0)
z = ZhangSuen(image)
rotatecoords = [(120.1,300.5,3), (300, 300.5,3), (112,300.5,22)]
z.myfunction(rotatecoords, image)