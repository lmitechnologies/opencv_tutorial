import numpy as np
import cv2 as cv
import argparse as arg
import imutils

ap = arg.ArgumentParser()
ap. add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

image = cv.imread(args["image"])
cv.imshow("Original", image)

r = 150.0 / image.shape[1]
dim = (150, int(image.shape[0] * r))

resized = cv.resize(image, dim, interpolation = cv.INTER_AREA)
cv.imshow("Resized (Width)", resized)

r = 50.0 / image.shape[0]
dim = (int(image.shape[1] * r), 50)

resized = cv.resize(image, dim, interpolation = cv.INTER_AREA)
cv.imshow("Resized (Height)", resized)

resized = imutils.resize(image, width = 100)
cv.imshow("Resized via Function", resized)
cv.waitKey(0)
