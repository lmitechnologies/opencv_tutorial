import numpy as np
import cv2 as cv
import argparse as arg
import imutils

ap = arg.ArgumentParser()
ap. add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

image = cv.imread(args["image"])
cv.imshow("Original", image)

(h, w) = image.shape[:2]
center = (w // 2, h // 2)

M = cv.getRotationMatrix2D(center, 45, 1.0)
rotated = cv.warpAffine(image, M, (w, h))
cv.imshow("Rotated by 45 Degrees", rotated)

M = cv.getRotationMatrix2D(center, -90, 1.0)
rotated = cv.warpAffine(image, M, (w, h))
cv.imshow("Rotated by -90 Degrees", rotated)

rotated = imutils.rotate(image, 180)
cv.imshow("Rotated by 180 Degrees", rotated)
cv.waitKey(0)
