import numpy as np
import cv2 as cv

def translate(image, x, y):
    M = np.float32([[1, 0, x],[0, 1, y]])
    shifted = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifted

def rotate(image, angle, center = None, scale = 1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w // 2, h // 2)
    
    M = cv.getRotationMatrix2D(center, angle, scale)
    rotated = cv.warpAffine(image, M, (w, h))
    return rotated
def resized(image, width = None, height = None, inter = cv.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim(width, int(h * r))
    
    resized = cv.resize(image, dim, interpolation= inter)
    return resized