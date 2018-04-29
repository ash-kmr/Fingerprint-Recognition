import sys
import numpy as np
import cv2
from zhangsuen import ZhangSuen
import math
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import utils

def gaborKernel(size, angle, frequency):
    """
    Create a Gabor kernel given a size, angle and frequency.

    Code is taken from https://github.com/rtshadow/biometrics.git
    """

    angle += np.pi * 0.5
    cos = np.cos(angle)
    sin = -np.sin(angle)

    yangle = lambda x, y: x * cos + y * sin
    xangle = lambda x, y: -x * sin + y * cos

    xsigma = ysigma = 4

    return utils.kernelFromFunction(size, lambda x, y:
            np.exp(-(
                (xangle(x, y) ** 2) / (xsigma ** 2) +
                (yangle(x, y) ** 2) / (ysigma ** 2)) / 2) *
            np.cos(2 * np.pi * frequency * xangle(x, y)))

def gaborFilter(image, orientations, frequencies, w=32):
    result = np.empty(image.shape)

    height, width = image.shape
    for y in range(0, height - w, w):
        for x in range(0, width - w, w):
            orientation = orientations[y+w//2, x+w//2]
            frequency = utils.averageFrequency(frequencies[y:y+w, x:x+w])

            if frequency < 0.0:
                result[y:y+w, x:x+w] = image[y:y+w, x:x+w]
                continue

            kernel = gaborKernel(16, orientation, frequency)
            result[y:y+w, x:x+w] = utils.convolve(image, kernel, (y, x), (w, w))

    return utils.normalize(result)


def gaborFilterSubdivide(image, orientations, frequencies, rect=None):
    if rect:
        y, x, h, w = rect
    else:
        y, x = 0, 0
        h, w = image.shape

    result = np.empty((h, w))

    orientation, deviation = utils.averageOrientation(
            orientations[y:y+h, x:x+w], deviation=True)

    if (deviation < 0.2 and h < 50 and w < 50) or h < 6 or w < 6:
        #print(deviation)
        #print(rect)

        frequency = utils.averageFrequency(frequencies[y:y+h, x:x+w])

        if frequency < 0.0:
            result = image[y:y+h, x:x+w]
        else:
            kernel = gaborKernel(16, orientation, frequency)
            result = utils.convolve(image, kernel, (y, x), (h, w))

    else:
        if h > w:
            hh = h // 2

            result[0:hh, 0:w] = \
                    gaborFilterSubdivide(image, orientations, frequencies, (y, x, hh, w))

            result[hh:h, 0:w] = \
                    gaborFilterSubdivide(image, orientations, frequencies, (y + hh, x, h - hh, w))
        else:
            hw = w // 2

            result[0:h, 0:hw] = \
                    gaborFilterSubdivide(image, orientations, frequencies, (y, x, h, hw))

            result[0:h, hw:w] = \
                    gaborFilterSubdivide(image, orientations, frequencies, (y, x + hw, h, w - hw))



    if w > 20 and h > 20:
        result = utils.normalize(result)

    return result

