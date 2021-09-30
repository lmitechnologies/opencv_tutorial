import numpy as np
import argparse as arg
import cv2 as cv

ap = arg.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

image = cv.imread(args["image"])
image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
blurred = cv.GaussianBlur(image, (5, 5), 0)
cv.imshow("Image", image)

(T, thresh) = cv.threshold(blurred, 155, 255, cv.THRESH_BINARY)
cv.imshow("Threshold Binary", thresh)

(T, threshInv) = cv.threshold(blurred, 155, 255, cv.THRESH_BINARY_INV)
cv.imshow("Threshold Binary Inverse", threshInv)
cv.imshow("Coins", cv.bitwise_and(image, image, mask = threshInv))
cv.waitKey(0)