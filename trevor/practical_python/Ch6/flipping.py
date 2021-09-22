import numpy as np
import cv2 as cv
import argparse as arg
import imutils

ap = arg.ArgumentParser()
ap. add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

image = cv.imread(args["image"])
cv.imshow("Original", image)

flipped = cv.flip(image, 1)
cv.imshow("Flipped Horizontally", flipped)

flipped = cv.flip(image, 0)
cv.imshow("Flipped Vertically", flipped)

flipped = cv.flip(image, -1)
cv.imshow("Flipped Horizontally & Vertically", flipped)


cv.waitKey(0)