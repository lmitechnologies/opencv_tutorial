import numpy as np
import argparse as arg
import cv2 as cv

ap = arg.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

image = cv.imread(args["image"])
image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

eq = cv.equalizeHist(image)

cv.imshow("Histogram Equalization", np.hstack([image, eq]))
cv.waitKey(0)