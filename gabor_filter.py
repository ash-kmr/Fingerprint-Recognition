import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
import scipy.signal as signal
import cv2


def apply_kernel_at(get_value, kernel, i, j):
    kernel = list(kernel)
    kernel_size = len(kernel)
    result = 0
    for k in range(0, kernel_size):
        for l in range(0, kernel_size):
            pixel = get_value(i + k - kernel_size // 2, j + l - kernel_size // 2)
            result += pixel * kernel[k][l]
    return result

def gabor_kernel(W, angle, freq):
    cos = math.cos(angle)
    sin = math.sin(angle)

    yangle = lambda x, y: x * cos + y * sin
    xangle = lambda x, y: -x * sin + y * cos

    xsigma = ysigma = 4

    return utils.kernel_from_function(W, lambda x, y:
        math.exp(-(
            (xangle(x, y) ** 2) / (xsigma ** 2) +
            (yangle(x, y) ** 2) / (ysigma ** 2)) / 2) *
        math.cos(2 * math.pi * freq * xangle(x, y)))

def gabor(im, W, angles):
    (x, y) = im.size
    im_load = im.load()

    freqs = frequency.freq(im, W, angles)
    print "computing local ridge frequency done"

    freqs = cv2.GaussianBlur(freqs,(3,3),0)

    for i in range(1, x / W - 1):
        for j in range(1, y / W - 1):
            kernel = gabor_kernel(W, angles[i][j], freqs[i][j])
            for k in range(0, W):
                for l in range(0, W):
                    im_load[i * W + k, j * W + l] = apply_kernel_at(
                        lambda x, y: im_load[x, y],
                        kernel,
                        i * W + k,
                        j * W + l)

    return im