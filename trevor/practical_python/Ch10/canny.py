import numpy as np
import argparse as arg
import cv2 as cv

ap = arg.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

image = cv.imread(args["image"])
image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
image = cv.GaussianBlur(image, (5,5), 0)
cv.imshow("Blurred", image)

canny = cv.Canny(image, 30, 150)
cv.imshow("Canny", canny)
cv.waitKey(0)